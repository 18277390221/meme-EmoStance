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
    import runpy
    runpy.run_path(str(Path(__file__).with_name("bootstrap.py")))["register_package"]()
    __package__="model_information_qwen_v3"

from communication.common import (
    AUTH_TIMEOUT,FRAME_LIMIT,Peer,ProtocolError,load_token,log,
    port_number,read_message,token_argument,
)
from .protocol import build_result
from .errors import PromptTooLong
from .sessions import BACKGROUND_ROLL_INTERVAL,BusyError,DialogueManager
from .request_cache import RequestCache
from .dialogue_log import append_dialogue,is_dialogue_line

MAX_PENDING_PER_CONNECTION=8
MAX_ACTIVE_SESSIONS=128
RECORDING_LEASE=30.
DEFAULT_MAX_TURNS=4


def log_message(direction,device,request_id,payload):
    """Print readable dialogue without dumping authentication/transport frames."""
    timestamp=datetime.now().astimezone().isoformat(timespec="seconds")
    detail={"device":device,"request_id":request_id,**payload}
    # JSON keeps Chinese readable and escapes newlines/control characters,
    # so simultaneous requests remain identifiable on one terminal line.
    line=f"[{timestamp}] [{direction}] {json.dumps(detail,ensure_ascii=False)}"
    log(line)
    if is_dialogue_line(line) and not append_dialogue(line):
        log("[对话日志] 写入失败，当前对话继续处理")


@dataclass
class Connection:
    name: str
    peer: Peer
    session_id: str
    resumable: bool
    protocol_version: int=1
    pending_ids: set[str]=field(default_factory=set)
    responses: set[asyncio.Task]=field(default_factory=set)


class InferenceServer:
    """Admit each user turn at network arrival; inference runs off the loop.

    An optional random hello.session_id survives reconnects. Legacy peers get a
    connection UUID, so equal display names never share dialogue histories.
    """
    def __init__(self,engine,token,*,manager=None,name="meme-model",max_clients=16,model_version=None):
        if not isinstance(token,str) or len(token)<16:
            raise ValueError("通信口令至少需要 16 个字符")
        if max_clients<1:
            raise ValueError("max_clients 必须为正数")
        self.engine=engine
        self._token=token
        self.manager=manager if manager is not None else DialogueManager(engine,max_turns=DEFAULT_MAX_TURNS)
        self.name=name
        self.max_clients=max_clients
        self.clients: dict[str,Connection]={}
        self.connections: set[asyncio.Task]=set()
        self.stop=asyncio.Event()
        self._server=None
        self._closing=False
        self.server_instance_id=uuid.uuid4().hex
        config=getattr(engine,"config",{})
        inferred_version=Path(config["run_dir"]).name if isinstance(config,dict) and config.get("run_dir") else "unknown"
        self.model_version=model_version or getattr(engine,"model_version",inferred_version)
        self.request_cache=RequestCache(clock=self.manager.clock)
        self._recording_until: dict[str,float]={}
        self._completion_tasks: set[asyncio.Task]=set()
        self._maintenance=None

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
        self._maintenance=asyncio.create_task(self._maintain_v2(),name="model-information-request-cache")
        return self

    def _expire_v2(self):
        # The manager itself resumes at the lease deadline, even if this
        # cleanup runs late. Repeated completion never extends user idle time.
        for receipt in self.request_cache.expire():
            self.manager.resume_idle(receipt.session_id)
        now=self.manager.clock()
        for session_id,deadline in list(self._recording_until.items()):
            if now>=deadline:
                self._recording_until.pop(session_id)
                self.manager.resume_idle(session_id)

    async def _maintain_v2(self):
        while True:
            await asyncio.sleep(.25)
            self._expire_v2()

    async def _complete_v2(self,receipt,future,device):
        """Own inference independently of any connection and cache before send."""
        try:
            result=await asyncio.shield(future)
            response=build_result(receipt.request_id,result)
            # The worker may have kept a detached old dialogue or applied an
            # repeated-reply rollover. Use that request's final context,
            # never the session's current conversation at delivery time.
            response["conversation_id"]=future.conversation_id
            response["background_revision"]=future.background_revision
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log_message("推理失败",device,receipt.request_id,{"error":type(exc).__name__})
            if isinstance(exc,PromptTooLong):
                response={"type":"error","id":receipt.request_id,"code":"input_too_long",
                    "message":"这次输入太长，请分成较短的一句话再发送"}
            else:
                response={"type":"error","id":receipt.request_id,"code":"handler_error",
                    "message":"模型处理失败，本轮未写入有效历史，请重试"}
        self.request_cache.finish(receipt,response)
        if response["type"]=="result":
            self.manager.hold_idle(receipt.session_id,self.request_cache.playback_lease)
        else:
            self.manager.resume_idle(receipt.session_id)
        self.request_cache.expire()

    async def _deliver_v2(self,connection,receipt):
        try:
            response=await asyncio.shield(receipt.completed)
            await connection.peer.send(response)
            if response["type"]=="result":
                payload={"reply":response["reply"],"meme_id":response["meme_id"]}
            else:
                payload={"code":response["code"],"message":response["message"]}
            log_message("服务器 → 树莓派",connection.name,receipt.request_id,payload)
        except (OSError,ConnectionError,asyncio.TimeoutError,ProtocolError) as exc:
            log_message("发送失败",connection.name,receipt.request_id,{"error":type(exc).__name__})
            connection.peer.writer.close()
        finally:
            connection.pending_ids.discard(receipt.request_id)

    async def _receipt_v2(self,connection,receipt):
        if receipt.response is not None:
            response=receipt.response
            await connection.peer.send(response)
            if response["type"]=="result":
                payload={"reply":response["reply"],"meme_id":response["meme_id"]}
            else:
                payload={"code":response["code"],"message":response["message"]}
            log_message("服务器 → 树莓派",connection.name,receipt.request_id,payload)
            return
        await connection.peer.send({"type":"accepted","id":receipt.request_id,"status":"processing"})
        if receipt.request_id not in connection.pending_ids:
            connection.pending_ids.add(receipt.request_id)
            task=asyncio.create_task(self._deliver_v2(connection,receipt))
            connection.responses.add(task)
            task.add_done_callback(connection.responses.discard)

    async def _v2_message(self,connection,message):
        self._expire_v2()
        kind=message["type"]
        session_id=connection.session_id
        peer=connection.peer
        request_id=message.get("id")
        active=self.request_cache.active.get(session_id)
        if kind in ("speech_start","speech_cancel"):
            if kind=="speech_start":
                if active is not None or (session_id not in self._recording_until and
                        len(self.request_cache.active)+len(self._recording_until)>=MAX_ACTIVE_SESSIONS):
                    await self._error(peer,"busy","当前仍在处理或播放，请稍后说话",device=connection.name)
                    return
                self.manager.hold_idle(session_id,RECORDING_LEASE)
                self._recording_until[session_id]=self.manager.clock()+RECORDING_LEASE
            elif self._recording_until.pop(session_id,None) is not None and active is None:
                self.manager.resume_idle(session_id)
            await peer.send({"type":"control_ack","control":kind})
            return
        if kind not in ("request","request_status","playback_done","playback_failed"):
            await self._error(peer,"bad_type","v2 消息类型无效",device=connection.name)
            return
        if not isinstance(request_id,str) or not 1<=len(request_id)<=128:
            await self._error(peer,"bad_request","消息需要 1 到 128 字符的字符串 id",device=connection.name)
            return
        receipt=self.request_cache.get(session_id,request_id)
        if kind in ("playback_done","playback_failed"):
            if receipt is None:
                await self._error(peer,"unknown_request","未找到该请求，请开始新的提问",request_id,device=connection.name)
                return
            if receipt.response is None:
                await self._error(peer,"not_ready","该请求仍在处理，尚未开始播放",request_id,device=connection.name)
                return
            if self.request_cache.release(receipt):
                self.manager.resume_idle(session_id)
            await peer.send({"type":"control_ack","control":kind,"id":request_id})
            return
        if kind=="request_status":
            if receipt is None:
                await self._error(peer,"unknown_request","本次服务启动中没有该请求记录，请重新提问",request_id,device=connection.name)
            else:
                await self._receipt_v2(connection,receipt)
            return
        text=message.get("text")
        if not isinstance(text,str) or not text.strip() or len(text)>16000:
            await self._error(peer,"bad_request","text 须为 1 到 16000 字符的非空字符串",request_id,device=connection.name)
            return
        if message.get("end") is not True:
            await self._error(peer,"incomplete_request","需要 end:true 表示本句已经发送完整",request_id,device=connection.name)
            return
        if receipt is not None:
            if receipt.text!=text:
                await self._error(peer,"request_conflict","相同请求编号不能对应不同文字",request_id,device=connection.name)
            else:
                await self._receipt_v2(connection,receipt)
            return
        if active is not None or len(self.request_cache.active)>=MAX_ACTIVE_SESSIONS:
            await self._error(peer,"busy","当前仍在处理或播放，请等待本轮结束",request_id,device=connection.name)
            return
        try:
            future=self.manager.accept(session_id,text)
        except BusyError:
            await self._error(peer,"busy","请求队列已满，请稍后重新发送",request_id,device=connection.name)
            return
        # Registration and idle hold are synchronous: a second connection with
        # this session cannot sneak in while the first acknowledgement drains.
        receipt=self.request_cache.register(session_id,request_id,text)
        self._recording_until.pop(session_id,None)
        self.manager.hold_idle(session_id)
        task=asyncio.create_task(self._complete_v2(receipt,future,connection.name))
        self._completion_tasks.add(task)
        task.add_done_callback(self._completion_tasks.discard)
        log_message("树莓派 → 服务器",connection.name,request_id,{"text":text,"end":True})
        # A cached completion can happen while send drains, but delivery starts
        # only after this independent accepted frame has been sent.
        await peer.send({"type":"accepted","id":request_id,"status":"processing"})
        if request_id not in connection.pending_ids:
            connection.pending_ids.add(request_id)
            response=asyncio.create_task(self._deliver_v2(connection,receipt))
            connection.responses.add(response)
            response.add_done_callback(connection.responses.discard)

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
            version=hello.get("protocol_version",1)
            if isinstance(version,bool) or not isinstance(version,int) or version not in (1,2):
                await self._error(peer,"unsupported_protocol","仅支持 protocol_version 1 或 2")
                return
            if version==2 and not resumable:
                await self._error(peer,"bad_request","v2 握手必须提供稳定 session_id")
                return
            if any(c.session_id==session_id and c.protocol_version!=version for c in self.clients.values()) or (
                    version==1 and any(r.session_id==session_id for r in self.request_cache.receipts.values())):
                await self._error(peer,"protocol_conflict","同一会话不能混用旧协议与 v2 协议")
                return
            if len(self.clients)>=self.max_clients:
                await self._error(peer,"busy","连接数已达上限")
                return
            connection=Connection(name,peer,session_id,resumable,version)
            self.clients[key]=connection
            response={"type":"hello_ok","name":self.name,"client_id":key,"session_id":session_id}
            if version==2:
                response.update({"protocol_version":2,"model_ready":True,"model_version":self.model_version,
                    "server_instance_id":self.server_instance_id,"max_inflight":1,
                    "idle_timeout":self.manager.idle_timeout,"max_turns":self.manager.max_turns,
                    "context_window_turns":self.manager.max_turns,"background_roll_interval":BACKGROUND_ROLL_INTERVAL,
                    "capabilities":["end","accepted","request_status","speech_start","speech_cancel",
                        "playback_done","playback_failed","conversation_id","sliding_context","background_revision"]})
            await peer.send(response)
            log(f"[已连接] {name} / {key[:8]}")
            while True:
                message=await read_message(reader)
                kind=message["type"]
                if kind=="ping":
                    # Transport traffic never counts as a user utterance.
                    await peer.send({"type":"pong"})
                    continue
                if connection.protocol_version==2:
                    await self._v2_message(connection,message)
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
        connections=list(self.connections)
        for task in connections:
            task.cancel()
        await asyncio.gather(*connections,return_exceptions=True)
        if self._server:
            # Python 3.12 waits for accepted transports as well as the listener.
            # Cancel handlers first so their finally blocks close those peers.
            await self._server.wait_closed()
        background=list(self._completion_tasks)
        if self._maintenance is not None:
            background.append(self._maintenance)
        for task in background:
            task.cancel()
        await asyncio.gather(*background,return_exceptions=True)
        await self.manager.close()
        self.request_cache.close()
        self._recording_until.clear()

    async def run(self,host="127.0.0.1",port=9000):
        await self.start(host,port)
        loop=asyncio.get_running_loop()
        for sig in (signal.SIGINT,signal.SIGTERM):
            try:
                loop.add_signal_handler(sig,self.stop.set)
            except (NotImplementedError,RuntimeError):
                pass
        turn_policy=("不限制历史轮数" if self.manager.max_turns is None else
            f"滚动保留最近 {self.manager.max_turns} 轮，背景每滑动 {BACKGROUND_ROLL_INTERVAL} 轮更新")
        log(f"模型通信服务已启动：{self.addresses}；用户空闲 {self.manager.idle_timeout:g} 秒后重置会话；"
            f"{turn_policy}")
        try:
            await self.stop.wait()
        finally:
            await self.close()
            log("模型通信服务已停止")


def main(argv=None):
    parser=argparse.ArgumentParser(description="树莓派文字 → EmoStance 推理 → 回复与表情编号")
    parser.add_argument("--host",default="127.0.0.1",help="服务器监听地址；跨设备使用服务器的 ZeroTier IP")
    parser.add_argument("--port",type=port_number,default=9000)
    parser.add_argument("--model-dir",type=Path,default=None,help="覆盖配置中的已训练 Qwen run_dir")
    parser.add_argument("--config",type=Path,default=None,help="默认 model_qwen/inference_config.json")
    parser.add_argument("--chat-config",type=Path,default=None,help="可选聊天参数 JSON")
    parser.add_argument("--device",choices=["auto","cuda","cpu"],default=None)
    parser.add_argument("--dtype",choices=["auto","bfloat16","float16","float32"],default=None)
    token_argument(parser)
    args=parser.parse_args(argv)
    try:
        token=load_token(args.token_file)
        from .model import load_chat_engine
        log("正在加载本地模型参数与回复检索库……")
        engine=load_chat_engine(args.model_dir,device=args.device,dtype=args.dtype,
                               config_path=args.config,settings_path=args.chat_config)
        asyncio.run(InferenceServer(engine,token).run(args.host,args.port))
    except (ValueError,OSError,RuntimeError,ImportError) as exc:
        parser.exit(1,f"启动失败：{exc}\n")
    except KeyboardInterrupt:
        pass


if __name__=="__main__":
    main()
