"""Business TCP server using the existing communication wire protocol."""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass,field
from datetime import datetime
import hmac
import json
from pathlib import Path
import signal
import sys
import uuid

if __package__ in (None,""):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from communication.common import (
    AUTH_TIMEOUT,FRAME_LIMIT,Peer,ProtocolError,load_token,log,
    port_number,read_message,token_argument,
)
from model_information.protocol import build_result
from model_information.errors import PromptTooLong
from model_information.sessions import BusyError,DialogueManager

MAX_PENDING_PER_CONNECTION=8


def log_message(direction,device,request_id,payload):
    """Print readable dialogue without dumping authentication/transport frames."""
    timestamp=datetime.now().astimezone().isoformat(timespec="seconds")
    detail={"device":device,"request_id":request_id,**payload}
    # JSON keeps Chinese readable and escapes newlines/control characters,
    # so simultaneous requests remain identifiable on one terminal line.
    log(f"[{timestamp}] [{direction}] {json.dumps(detail,ensure_ascii=False)}")


@dataclass
class Connection:
    name: str
    peer: Peer
    session_id: str
    resumable: bool
    pending_ids: set[str]=field(default_factory=set)
    responses: set[asyncio.Task]=field(default_factory=set)


class InferenceServer:
    """Admit each user turn at network arrival; inference runs off the loop.

    An optional random hello.session_id survives reconnects. Legacy peers get a
    connection UUID, so equal display names never share dialogue histories.
    """
    def __init__(self,engine,token,*,manager=None,name="meme-model",max_clients=16):
        if not isinstance(token,str) or len(token)<16:
            raise ValueError("通信口令至少需要 16 个字符")
        if max_clients<1:
            raise ValueError("max_clients 必须为正数")
        self.engine=engine
        self._token=token
        self.manager=manager if manager is not None else DialogueManager(engine)
        self.name=name
        self.max_clients=max_clients
        self.clients: dict[str,Connection]={}
        self.connections: set[asyncio.Task]=set()
        self.stop=asyncio.Event()
        self._server=None
        self._closing=False

    @property
    def addresses(self):
        return [sock.getsockname() for sock in self._server.sockets] if self._server else []

    async def start(self,host="127.0.0.1",port=9000):
        if self._closing:
            raise RuntimeError("服务已经关闭")
        if self._server is not None:
            return self
        self.manager.start()
        try:
            self._server=await asyncio.start_server(self.connected,host,port,limit=FRAME_LIMIT)
        except BaseException:
            await self.manager.close()
            raise
        return self

    async def _error(self,peer,code,message,request_id=None,*,device="raspberrypi"):
        response={"type":"error","code":code,"message":message}
        if request_id is not None:
            response["id"]=request_id
        await peer.send(response)
        if request_id is not None:
            log_message("服务器 → 树莓派",device,request_id,{"code":code,"message":message})

    async def _respond(self,connection,request_id,future):
        try:
            try:
                result=await asyncio.shield(future)
                response=build_result(request_id,result)
            except Exception as exc:
                # Exception details and authentication tokens are not dialogue.
                log_message("推理失败",connection.name,request_id,{"error":type(exc).__name__})
                if isinstance(exc,PromptTooLong):
                    response={"type":"error","id":request_id,"code":"input_too_long",
                        "message":"这次输入太长，请分成较短的一句话再发送"}
                else:
                    response={"type":"error","id":request_id,"code":"handler_error",
                        "message":"模型处理失败，本轮未写入有效历史，请重试"}
            try:
                await connection.peer.send(response)
                if response["type"]=="result":
                    log_message("服务器 → 树莓派",connection.name,request_id,
                        {"reply":response["reply"],"meme_id":response["meme_id"]})
                else:
                    log_message("服务器 → 树莓派",connection.name,request_id,
                        {"code":response["code"],"message":response["message"]})
            except (OSError,ConnectionError,asyncio.TimeoutError,ProtocolError) as exc:
                log_message("发送失败",connection.name,request_id,{"error":type(exc).__name__})
                connection.peer.writer.close()
        finally:
            connection.pending_ids.discard(request_id)

    async def connected(self,reader,writer):
        task=asyncio.current_task()
        self.connections.add(task)
        peer=Peer(writer)
        key=uuid.uuid4().hex
        connection=None
        try:
            if self._closing:
                return
            hello=await read_message(reader,AUTH_TIMEOUT)
            supplied=hello.get("token")
            if hello["type"]!="hello" or not isinstance(supplied,str) or not hmac.compare_digest(
                supplied.encode("utf-8"),self._token.encode("utf-8")):
                await self._error(peer,"auth_failed","通信口令错误")
                return
            name=hello.get("name","raspberrypi")
            if not isinstance(name,str) or not 1<=len(name)<=64 or not name.isprintable():
                raise ProtocolError("客户端名称须为 1 到 64 个可打印字符")
            resumable="session_id" in hello
            session_id=hello.get("session_id",key)
            if not isinstance(session_id,str) or not session_id.strip() or len(session_id)>128 or not session_id.isprintable():
                raise ProtocolError("session_id 须为 1 到 128 个可打印字符")
            if len(self.clients)>=self.max_clients:
                await self._error(peer,"busy","连接数已达上限")
                return
            connection=Connection(name,peer,session_id,resumable)
            self.clients[key]=connection
            await peer.send({"type":"hello_ok","name":self.name,"client_id":key,"session_id":session_id})
            log(f"[已连接] {name} / {key[:8]}")
            while True:
                message=await read_message(reader)
                kind=message["type"]
                if kind=="ping":
                    # Transport traffic never counts as a user utterance.
                    await peer.send({"type":"pong"})
                    continue
                if kind!="request":
                    await self._error(peer,"bad_type","仅支持 request 或 ping")
                    continue
                request_id,text=message.get("id"),message.get("text")
                if not isinstance(request_id,str) or not 1<=len(request_id)<=128:
                    await self._error(peer,"bad_request","request 需要 1 到 128 字符的字符串 id")
                    continue
                if not isinstance(text,str) or not text.strip() or len(text)>16000:
                    await self._error(peer,"bad_request","text 须为 1 到 16000 字符的非空字符串",request_id,device=name)
                    continue
                if request_id in connection.pending_ids:
                    log_message("树莓派 → 服务器",name,request_id,{"text":text,"status":"rejected"})
                    await self._error(peer,"duplicate_request","该请求编号仍在处理中",request_id,device=name)
                    continue
                if len(connection.pending_ids)>=MAX_PENDING_PER_CONNECTION:
                    log_message("树莓派 → 服务器",name,request_id,{"text":text,"status":"rejected"})
                    await self._error(peer,"busy","已有 8 条请求等待返回，请稍后重新发送",request_id,device=name)
                    continue
                try:
                    # No await before this registration: capture the user turn
                    # now, before any device/model queue delays its execution.
                    future=self.manager.accept(session_id,text)
                except BusyError:
                    log_message("树莓派 → 服务器",name,request_id,{"text":text,"status":"rejected"})
                    await self._error(peer,"busy","请求队列已满，请稍后重新发送",request_id,device=name)
                    continue
                except (ValueError,TypeError) as exc:
                    log_message("树莓派 → 服务器",name,request_id,{"text":text,"status":"rejected"})
                    await self._error(peer,"bad_request",str(exc),request_id,device=name)
                    continue
                # Register arrival first: terminal output must not postpone
                # the 10-second dialogue deadline or the request's queue slot.
                log_message("树莓派 → 服务器",connection.name,request_id,{"text":text})
                connection.pending_ids.add(request_id)
                response=asyncio.create_task(self._respond(connection,request_id,future))
                connection.responses.add(response)
                response.add_done_callback(connection.responses.discard)
        except EOFError:
            pass
        except ProtocolError as exc:
            try:
                await self._error(peer,"protocol_error",str(exc))
            except (OSError,ConnectionError,asyncio.TimeoutError,ProtocolError):
                pass
        except (OSError,ConnectionError,asyncio.TimeoutError):
            pass
        finally:
            self.clients.pop(key,None)
            if connection:
                responses=list(connection.responses)
                for response in responses:
                    response.cancel()
                await asyncio.gather(*responses,return_exceptions=True)
                if not connection.resumable:
                    self.manager.reset(connection.session_id)
            await peer.close()
            self.connections.discard(task)

    async def close(self):
        if self._closing:
            return
        self._closing=True
        self.stop.set()
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        connections=list(self.connections)
        for task in connections:
            task.cancel()
        await asyncio.gather(*connections,return_exceptions=True)
        await self.manager.close()

    async def run(self,host="127.0.0.1",port=9000):
        await self.start(host,port)
        loop=asyncio.get_running_loop()
        for sig in (signal.SIGINT,signal.SIGTERM):
            try:
                loop.add_signal_handler(sig,self.stop.set)
            except (NotImplementedError,RuntimeError):
                pass
        log(f"模型通信服务已启动：{self.addresses}；用户空闲 {self.manager.idle_timeout:g} 秒后重置会话；"
            f"每段最多 {self.manager.max_turns} 轮，下一条消息开启新对话")
        try:
            await self.stop.wait()
        finally:
            await self.close()
            log("模型通信服务已停止")


def main():
    parser=argparse.ArgumentParser(description="树莓派文字 → EmoStance 推理 → 回复与表情编号")
    parser.add_argument("--host",default="127.0.0.1",help="服务器监听地址；跨设备使用服务器的 ZeroTier IP")
    parser.add_argument("--port",type=port_number,default=9000)
    parser.add_argument("--model-dir",type=Path,default=None,help="默认 model_canshu/V2.0.0")
    parser.add_argument("--device",choices=["auto","cuda","cpu"],default=None)
    parser.add_argument("--dtype",choices=["auto","bfloat16","float16","float32"],default=None)
    token_argument(parser)
    args=parser.parse_args()
    try:
        token=load_token(args.token_file)
        from model_information.model import load_chat_engine
        log("正在加载本地模型参数与回复检索库……")
        engine=load_chat_engine(args.model_dir,device=args.device,dtype=args.dtype)
        asyncio.run(InferenceServer(engine,token).run(args.host,args.port))
    except (ValueError,OSError,RuntimeError,ImportError) as exc:
        parser.exit(1,f"启动失败：{exc}\n")
    except KeyboardInterrupt:
        pass


if __name__=="__main__":
    main()
