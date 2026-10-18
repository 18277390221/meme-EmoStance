"""The production handshake on real TCP sockets, using a CPU fake engine."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from communication.common import FRAME_LIMIT, Peer, read_message
from model_information_qwen_v3.server import InferenceServer
from model_information_qwen_v3.sessions import DialogueManager
from model_information_qwen_v3.tests import test_server as legacy


class V2ServerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.clock=legacy.FakeClock()
        self.engine=legacy.RecordingEngine()
        self.server=None
        self.peers=[]
        self.logs=[]
        self.log_patch=patch("model_information_qwen_v3.server.log",side_effect=self.logs.append)
        self.log_patch.start()
        journal=tempfile.TemporaryDirectory()
        self.addCleanup(journal.cleanup)
        journal_patch=patch("model_information_qwen_v3.dialogue_log.DEFAULT_LOG_PATH",Path(journal.name)/"conversation.log")
        journal_patch.start()
        self.addCleanup(journal_patch.stop)

    async def asyncTearDown(self):
        self.engine.release.set()
        for peer in self.peers:
            await peer.close()
        if self.server is not None:
            await self.server.close()
        self.log_patch.stop()

    async def launch(self, *, block_text=None, use_server_default=False, **options):
        self.engine.block_text=block_text
        if use_server_default:
            self.server=InferenceServer(self.engine,legacy.TOKEN,model_version="V2.0.0")
            self.server.manager.clock=self.clock
            self.server.request_cache.clock=self.clock
        else:
            manager=DialogueManager(self.engine,clock=self.clock,cleanup_interval=None,**options)
            self.server=InferenceServer(self.engine,legacy.TOKEN,manager=manager,model_version="V2.0.0")
        await self.server.start("127.0.0.1",0)
        self.port=self.server.addresses[0][1]
        return self.server

    async def connect(self, session_id="stable-random-session", **hello_options):
        reader,writer=await asyncio.open_connection("127.0.0.1",self.port,limit=FRAME_LIMIT)
        peer=Peer(writer)
        self.peers.append(peer)
        hello={"type":"hello","name":"meme-player","token":legacy.TOKEN,"protocol_version":2}
        if session_id is not None:
            hello["session_id"]=session_id
        hello.update(hello_options)
        await peer.send(hello)
        return reader,peer,await read_message(reader,1)

    async def ask(self, reader, peer, text="你好", request_id="one"):
        await peer.send({"type":"request","id":request_id,"text":text,"end":True})
        accepted=await read_message(reader,1)
        self.assertEqual(accepted,{"type":"accepted","id":request_id,"status":"processing"})
        return await read_message(reader,1)

    async def control(self, reader, peer, kind, request_id=None):
        frame={"type":kind}
        if request_id is not None:
            frame["id"]=request_id
        await peer.send(frame)
        return await read_message(reader,1)

    def sent_logs(self, request_id):
        marker="] [服务器 → 树莓派] "
        payloads=[json.loads(line.split(marker,1)[1]) for line in self.logs if marker in line]
        return [payload for payload in payloads if payload.get("request_id")==request_id]

    async def test_ready_handshake_and_new_boot_epoch(self):
        await self.launch()
        _,_,hello=await self.connect()
        self.assertEqual(hello["protocol_version"],2)
        self.assertTrue(hello["model_ready"])
        self.assertEqual(hello["model_version"],"V2.0.0")
        self.assertEqual(hello["max_inflight"],1)
        self.assertEqual(hello["max_turns"],4)
        self.assertEqual(hello["context_window_turns"],4)
        self.assertEqual(hello["background_roll_interval"],5)
        self.assertIn("request_status",hello["capabilities"])
        self.assertIn("conversation_id",hello["capabilities"])
        self.assertIn("sliding_context",hello["capabilities"])
        self.assertIn("background_revision",hello["capabilities"])
        self.assertEqual(hello["server_instance_id"],self.server.server_instance_id)
        another=InferenceServer(self.engine,legacy.TOKEN)
        self.assertNotEqual(another.server_instance_id,self.server.server_instance_id)
        await another.close()

    async def test_v2_requires_stable_session_and_valid_version(self):
        await self.launch()
        _,_,response=await self.connect(session_id=None)
        self.assertEqual(response["code"],"bad_request")
        for value in (True,2.0,3,"2"):
            _,_,response=await self.connect(protocol_version=value)
            self.assertEqual(response["code"],"unsupported_protocol")
        self.assertEqual(self.engine.calls,[])

    async def test_no_complete_end_flag_never_reaches_inference(self):
        await self.launch()
        reader,peer,_=await self.connect()
        for end in (None,False,0,1,"true"):
            frame={"type":"request","id":"partial","text":"未结束"}
            if end is not None:
                frame["end"]=end
            await peer.send(frame)
            response=await read_message(reader,1)
            self.assertEqual(response["code"],"incomplete_request")
        self.assertEqual(self.engine.calls,[])
        self.assertEqual(self.server.manager.pending_count,0)
        self.assertEqual(self.server.request_cache.receipts,{})
        self.assertEqual((await self.ask(reader,peer))["type"],"result")

    async def test_duplicate_processing_receipt_runs_and_commits_once(self):
        await self.launch(block_text="慢问题")
        reader,peer,_=await self.connect()
        frame={"type":"request","id":"same","text":"慢问题","end":True}
        await peer.send(frame)
        self.assertEqual((await read_message(reader,1))["type"],"accepted")
        await legacy.until(self.engine.started.is_set)
        await peer.send(frame)
        self.assertEqual((await read_message(reader,1))["type"],"accepted")
        self.assertEqual(self.server.manager.pending_count,1)
        self.engine.release.set()
        self.assertEqual((await read_message(reader,1))["type"],"result")
        await peer.send({"type":"ping"})
        self.assertEqual(await read_message(reader,1),{"type":"pong"})
        self.assertEqual(len(self.engine.calls),1)
        self.assertEqual(len(self.server.manager.history("stable-random-session")),2)

    async def test_duplicate_completed_receipt_and_conflict(self):
        await self.launch()
        reader,peer,_=await self.connect()
        original=await self.ask(reader,peer)
        await peer.send({"type":"request","id":"one","text":"你好","end":True})
        self.assertEqual(await read_message(reader,1),original)
        await peer.send({"type":"request","id":"one","text":"不同文字","end":True})
        self.assertEqual((await read_message(reader,1))["code"],"request_conflict")
        self.assertEqual(len(self.engine.calls),1)

    async def test_same_id_conflict_while_processing_never_submits_second(self):
        await self.launch(block_text="慢问题")
        reader,peer,_=await self.connect()
        await peer.send({"type":"request","id":"same","text":"慢问题","end":True})
        await read_message(reader,1)
        await legacy.until(self.engine.started.is_set)
        await peer.send({"type":"request","id":"same","text":"不同文字","end":True})
        self.assertEqual((await read_message(reader,1))["code"],"request_conflict")
        self.assertEqual(len(self.engine.calls),1)

    async def test_reconnect_queries_running_request_without_duplicate_commit(self):
        await self.launch(block_text="慢问题")
        reader,peer,hello=await self.connect()
        await peer.send({"type":"request","id":"recover","text":"慢问题","end":True})
        await read_message(reader,1)
        await legacy.until(self.engine.started.is_set)
        await peer.close()
        await legacy.until(lambda:not self.server.clients)
        self.clock.set(100)
        reader,peer,newhello=await self.connect()
        self.assertEqual(hello["server_instance_id"],newhello["server_instance_id"])
        await peer.send({"type":"request_status","id":"recover"})
        self.assertEqual((await read_message(reader,1))["type"],"accepted")
        self.engine.release.set()
        response=await read_message(reader,1)
        self.assertEqual(response["id"],"recover")
        self.assertEqual(response["type"],"result")
        self.assertEqual(len(self.engine.calls),1)
        self.assertEqual(len(self.server.manager.history("stable-random-session")),2)

    async def test_ack_send_failure_still_finishes_and_caches_request(self):
        await self.launch(block_text="慢问题")
        reader,peer,_=await self.connect()
        original_send=Peer.send

        async def fail_ack(target,message):
            if message.get("type")=="accepted":
                raise ConnectionError("test dropped accepted")
            await original_send(target,message)

        with patch("model_information_qwen_v3.server.Peer.send",new=fail_ack):
            await peer.send({"type":"request","id":"lost-ack","text":"慢问题","end":True})
            await legacy.until(self.engine.started.is_set)
            self.assertEqual(await asyncio.wait_for(reader.read(),1),b"")
        self.engine.release.set()
        await legacy.until(lambda:self.server.request_cache.get("stable-random-session","lost-ack").response is not None)
        self.assertEqual(self.sent_logs("lost-ack"),[])
        reader,peer,_=await self.connect()
        await peer.send({"type":"request_status","id":"lost-ack"})
        response=await read_message(reader,1)
        self.assertEqual(response["type"],"result")
        await legacy.until(lambda:len(self.sent_logs("lost-ack"))==1)
        self.assertEqual(self.sent_logs("lost-ack")[0],{
            "device":"meme-player","request_id":"lost-ack",
            "reply":"回答：慢问题","meme_id":"meme-0062"})
        self.assertEqual(len(self.engine.calls),1)
        self.assertEqual(len(self.server.manager.history("stable-random-session")),2)

    async def test_failed_original_result_send_is_logged_when_cache_recovery_succeeds(self):
        await self.launch()
        reader,peer,_=await self.connect()
        original_send=Peer.send

        async def fail_result(target,message):
            if message.get("type")=="result":
                raise ConnectionError("test dropped final result")
            await original_send(target,message)

        with patch("model_information_qwen_v3.server.Peer.send",new=fail_result):
            await peer.send({"type":"request","id":"lost-result","text":"你好","end":True})
            self.assertEqual((await read_message(reader,1))["type"],"accepted")
            self.assertEqual(await asyncio.wait_for(reader.read(),1),b"")
            await legacy.until(lambda:not self.server.clients)
        self.assertEqual(self.sent_logs("lost-result"),[])
        reader,peer,_=await self.connect()
        await peer.send({"type":"request_status","id":"lost-result"})
        response=await read_message(reader,1)
        self.assertEqual(response["reply"],"回答：你好")
        await legacy.until(lambda:len(self.sent_logs("lost-result"))==1)
        self.assertEqual(self.sent_logs("lost-result")[0],{
            "device":"meme-player","request_id":"lost-result",
            "reply":"回答：你好","meme_id":"meme-0062"})
        self.assertEqual(len(self.engine.calls),1)
        self.assertEqual(len(self.server.manager.history("stable-random-session")),2)

    async def test_final_result_is_recoverable_after_connection_disappears(self):
        await self.launch()
        reader,peer,_=await self.connect()
        result=await self.ask(reader,peer)
        self.assertRegex(result["conversation_id"],r"^[0-9a-f]{32}$")
        await peer.close()
        await legacy.until(lambda:not self.server.clients)
        reader,peer,_=await self.connect()
        await peer.send({"type":"request_status","id":"one"})
        self.assertEqual(await read_message(reader,1),result)
        await legacy.until(lambda:len(self.sent_logs("one"))==2)
        self.assertEqual(self.sent_logs("one")[0],self.sent_logs("one")[1])
        self.assertEqual(len(self.engine.calls),1)

    async def test_single_flight_covers_processing_and_waiting_for_playback(self):
        await self.launch(block_text="慢问题")
        reader,peer,_=await self.connect()
        await peer.send({"type":"request","id":"first","text":"慢问题","end":True})
        await read_message(reader,1)
        await legacy.until(self.engine.started.is_set)
        await peer.send({"type":"request","id":"second","text":"太早","end":True})
        self.assertEqual((await read_message(reader,1))["code"],"busy")
        self.assertEqual((await self.control(reader,peer,"playback_done","first"))["code"],"not_ready")
        self.engine.release.set()
        await read_message(reader,1)
        await peer.send({"type":"request","id":"second","text":"太早","end":True})
        self.assertEqual((await read_message(reader,1))["code"],"busy")
        self.assertEqual((await self.control(reader,peer,"playback_done","wrong-id"))["code"],"unknown_request")
        self.assertEqual((await self.control(reader,peer,"playback_done","first"))["type"],"control_ack")
        self.assertEqual((await self.ask(reader,peer,"下一句","second"))["type"],"result")
        self.assertEqual(len(self.engine.calls),2)

    async def test_same_session_two_connections_cannot_bypass_single_flight(self):
        await self.launch()
        r1,p1,_=await self.connect()
        r2,p2,_=await self.connect()
        await self.ask(r1,p1)
        await p2.send({"type":"request","id":"second","text":"绕过限额","end":True})
        self.assertEqual((await read_message(r2,1))["code"],"busy")
        await p1.close()
        await legacy.until(lambda:len(self.server.clients)==1)
        await p2.send({"type":"request_status","id":"one"})
        self.assertEqual((await read_message(r2,1))["type"],"result")
        self.assertEqual(len(self.engine.calls),1)

    async def test_v2_session_cannot_be_reopened_with_legacy_to_bypass_limit(self):
        await self.launch()
        reader,peer,_=await self.connect()
        await self.ask(reader,peer)
        _,_,hello=await self.connect(protocol_version=1)
        self.assertEqual(hello["code"],"protocol_conflict")

    async def test_result_keeps_history_until_playback_then_exactly_ten_idle_seconds(self):
        await self.launch()
        reader,peer,_=await self.connect()
        await self.ask(reader,peer)
        self.clock.set(100)
        self.assertEqual(len(self.server.manager.history("stable-random-session")),2)
        await self.control(reader,peer,"playback_done","one")
        self.clock.set(109)
        self.assertEqual(len(self.server.manager.history("stable-random-session")),2)
        # A duplicate playback completion is acknowledged but cannot postpone
        # expiry and keep a dialogue alive forever.
        await self.control(reader,peer,"playback_done","one")
        self.clock.set(110)
        self.assertEqual(self.server.manager.history("stable-random-session"),[])

    async def test_actual_speech_holds_idle_and_speech_cancel_resumes_deadline(self):
        await self.launch()
        reader,peer,_=await self.connect()
        await self.ask(reader,peer)
        await self.control(reader,peer,"playback_done","one")
        self.clock.set(9)
        self.assertEqual((await self.control(reader,peer,"speech_start"))["type"],"control_ack")
        self.clock.set(20)
        self.assertEqual(len(self.server.manager.history("stable-random-session")),2)
        await self.control(reader,peer,"speech_cancel")
        self.clock.set(29)
        self.assertEqual(len(self.server.manager.history("stable-random-session")),2)
        self.clock.set(30)
        self.assertEqual(self.server.manager.history("stable-random-session"),[])

    async def test_recording_and_playback_leases_recover_missing_callbacks(self):
        await self.launch()
        reader,peer,_=await self.connect()
        await self.ask(reader,peer)
        self.clock.set(129)
        self.assertEqual(len(self.server.manager.history("stable-random-session")),2)
        self.clock.set(130)
        self.assertEqual(self.server.manager.history("stable-random-session"),[])
        await self.ask(reader,peer,"重新开局","two")
        await self.control(reader,peer,"playback_failed","two")
        self.clock.set(139)
        await self.control(reader,peer,"speech_start")
        self.clock.set(178)
        self.assertEqual(len(self.server.manager.history("stable-random-session")),2)
        self.clock.set(179)
        self.assertEqual(self.server.manager.history("stable-random-session"),[])

    async def test_failures_are_cached_and_release_slot_without_counting_turn(self):
        await self.launch()
        reader,peer,_=await self.connect()
        error=await self.ask(reader,peer,"失败","failed")
        self.assertEqual(error["code"],"handler_error")
        await peer.send({"type":"request_status","id":"failed"})
        self.assertEqual(await read_message(reader,1),error)
        await legacy.until(lambda:len(self.sent_logs("failed"))==2)
        self.assertEqual(self.sent_logs("failed")[0],self.sent_logs("failed")[1])
        self.assertEqual(self.sent_logs("failed")[1]["code"],"handler_error")
        self.assertEqual(self.server.manager.history("stable-random-session"),[])
        self.assertEqual((await self.ask(reader,peer,"有效问题","next"))["type"],"result")
        self.assertEqual(len(self.engine.calls),2)

    async def test_unknown_request_and_cache_ttl_do_not_reinfer(self):
        await self.launch()
        reader,peer,_=await self.connect()
        await peer.send({"type":"request_status","id":"unknown"})
        self.assertEqual((await read_message(reader,1))["code"],"unknown_request")
        await self.ask(reader,peer)
        await self.control(reader,peer,"playback_done","one")
        self.clock.set(600)
        await peer.send({"type":"request_status","id":"one"})
        self.assertEqual((await read_message(reader,1))["code"],"unknown_request")
        self.assertEqual(len(self.engine.calls),1)

    async def test_injected_unlimited_manager_keeps_twelve_playback_rounds_in_one_conversation(self):
        await self.launch(max_turns=None)
        reader,peer,hello=await self.connect()
        self.assertIsNone(hello["max_turns"])
        expected=[]
        conversation_ids=[]
        for number in range(1,13):
            self.clock.set(number)
            request_id=f"turn-{number}"
            result=await self.ask(reader,peer,f"第{number}轮",request_id)
            self.assertEqual(result["type"],"result")
            conversation_ids.append(result["conversation_id"])
            expected.append({"role":"user","content":f"第{number}轮"})
            self.assertEqual(self.engine.calls[-1],expected)
            expected.append({"role":"assistant","content":f"回答：第{number}轮"})
            await self.control(reader,peer,"playback_done",request_id)
            self.assertEqual(self.server.manager.history("stable-random-session"),expected)
        self.assertEqual([len(call) for call in self.engine.calls],list(range(1,24,2)))
        self.assertEqual(len(self.server.manager.history("stable-random-session")),24)
        self.assertEqual(len(set(conversation_ids)),1)
        self.clock.set(22)
        self.assertEqual(self.server.manager.history("stable-random-session"),[])

    async def test_default_window_advances_background_at_nine_fourteen_and_preserves_cached_revision(self):
        await self.launch(use_server_default=True)
        reader,peer,hello=await self.connect()
        self.assertEqual(hello["max_turns"],4)
        other_reader,other_peer,_=await self.connect("separate-device-session")
        results=[]
        for number in range(1,17):
            self.clock.set(number/2)
            request_id=f"default-turn-{number}"
            result=await self.ask(reader,peer,f"默认第{number}轮",request_id)
            self.assertEqual(result["type"],"result")
            results.append(result)
            await self.control(reader,peer,"playback_done",request_id)
            if number==3:
                other=await self.ask(other_reader,other_peer,"另一会话","independent")
                await self.control(other_reader,other_peer,"playback_done","independent")
        own_calls=[call for call in self.engine.calls if call[-1]["content"].startswith("默认")]
        self.assertEqual([len(call) for call in own_calls],[1,3,5,7]+[9]*12)
        self.assertEqual([call[-1]["content"] for call in own_calls],[f"默认第{n}轮" for n in range(1,17)])
        ids=[result["conversation_id"] for result in results]
        self.assertEqual(len(set(ids)),1)
        self.assertEqual([result["background_revision"] for result in results],[0]*8+[1]*5+[2]*3)
        self.assertTrue(all(type(result["background_revision"]) is int for result in results))
        self.assertEqual(own_calls[4][0],{"role":"user","content":"默认第1轮"})
        self.assertNotIn(other["conversation_id"],ids)
        self.assertEqual(self.server.manager.history("stable-random-session"),[
            message for number in range(13,17) for message in (
                {"role":"user","content":f"默认第{number}轮"},
                {"role":"assistant","content":f"回答：默认第{number}轮"})])
        self.assertEqual(self.server.manager.history("separate-device-session"),[
            {"role":"user","content":"另一会话"},{"role":"assistant","content":"回答：另一会话"}])
        await peer.close()
        await legacy.until(lambda:len(self.server.clients)==1)
        reader,peer,_=await self.connect()
        await peer.send({"type":"request_status","id":"default-turn-1"})
        self.assertEqual(await read_message(reader,1),results[0])
        await peer.send({"type":"request_status","id":"default-turn-9"})
        self.assertEqual(await read_message(reader,1),results[8])
        self.assertEqual(len(self.engine.calls),17,"Cached lookup must not slide history or rerun the model")

    async def test_repeated_replies_roll_next_request_while_cache_retains_original_conversation(self):
        await self.launch(use_server_default=True)
        original_reply=self.engine.reply

        def reply(messages):
            result=original_reply(messages)
            text=messages[-1]["content"]
            if text in ("第一句", "第二句", "另一设备"):
                result["reply"]="同一句回复"
                result["meme_id"]="meme-0001" if text=="第一句" else "meme-0002"
            return result

        self.engine.reply=reply
        reader,peer,_=await self.connect()
        other_reader,other_peer,_=await self.connect("separate-device-session")
        first=await self.ask(reader,peer,"第一句","repeat-one")
        await self.control(reader,peer,"playback_done","repeat-one")
        other=await self.ask(other_reader,other_peer,"另一设备","independent")
        await self.control(other_reader,other_peer,"playback_done","independent")
        second=await self.ask(reader,peer,"第二句","repeat-two")
        self.assertEqual(first["reply"],second["reply"])
        self.assertNotEqual(first["meme_id"],second["meme_id"])
        self.assertEqual(first["conversation_id"],second["conversation_id"])
        self.assertEqual(first["background_revision"],0)
        self.assertEqual(second["background_revision"],0)
        await peer.send({"type":"request_status","id":"repeat-two"})
        self.assertEqual(await read_message(reader,1),second)
        self.assertEqual(len(self.engine.calls),3)
        self.assertEqual(len(self.server.manager.history("stable-random-session")),4)
        await self.control(reader,peer,"playback_done","repeat-two")
        third=await self.ask(reader,peer,"新问题","repeat-three")
        self.assertEqual(self.engine.calls[-1],[{"role":"user","content":"新问题"}])
        self.assertNotEqual(second["conversation_id"],third["conversation_id"])
        self.assertEqual(third["background_revision"],0)
        await self.control(reader,peer,"playback_done","repeat-three")
        other_second=await self.ask(other_reader,other_peer,"继续另一设备","independent-two")
        self.assertEqual(other["conversation_id"],other_second["conversation_id"])
        self.assertEqual([m["content"] for m in self.engine.calls[-1]],["另一设备","同一句回复","继续另一设备"])
        await self.control(other_reader,other_peer,"playback_done","independent-two")
        await peer.close()
        await legacy.until(lambda:len(self.server.clients)==1)
        reader,peer,_=await self.connect()
        await peer.send({"type":"request_status","id":"repeat-two"})
        self.assertEqual(await read_message(reader,1),second)
        await peer.send({"type":"request","id":"repeat-one","text":"第一句","end":True})
        self.assertEqual(await read_message(reader,1),first)
        self.assertEqual(len(self.engine.calls),5,"Cached lookup and retransmission must not generate or commit another reply")
        self.assertEqual(self.server.manager.history("stable-random-session"),[
            {"role":"user","content":"新问题"},{"role":"assistant","content":"回答：新问题"}])

    async def test_explicit_two_pair_window_still_supported(self):
        await self.launch(max_turns=2)
        reader,peer,hello=await self.connect()
        self.assertEqual(hello["max_turns"],2)
        self.assertEqual(hello["context_window_turns"],2)
        conversation_ids=[]
        for number in range(1,7):
            request_id=f"turn-{number}"
            result=await self.ask(reader,peer,f"第{number}轮",request_id)
            self.assertEqual(result["type"],"result")
            conversation_ids.append(result["conversation_id"])
            await self.control(reader,peer,"playback_done",request_id)
        self.assertEqual([len(call) for call in self.engine.calls],[1,3,5,5,5,5])
        self.assertEqual(len(self.server.manager.history("stable-random-session")),4)
        self.assertEqual(len(set(conversation_ids)),1)

    async def test_failed_request_and_cache_queries_do_not_advance_background_boundary(self):
        await self.launch(use_server_default=True)
        reader,peer,_=await self.connect()
        results=[]
        for number in range(1,9):
            result=await self.ask(reader,peer,f"问题{number}",f"window-{number}")
            results.append(result)
            await self.control(reader,peer,"playback_done",f"window-{number}")
        before=self.server.manager.history("stable-random-session")
        failed=await self.ask(reader,peer,"失败","failed-window")
        self.assertEqual(failed["type"],"error")
        self.assertNotIn("background_revision",failed)
        self.assertEqual(self.server.manager.history("stable-random-session"),before)
        await peer.send({"type":"request_status","id":"window-8"})
        self.assertEqual(await read_message(reader,1),results[-1])
        self.assertEqual(results[-1]["background_revision"],0)
        ninth=await self.ask(reader,peer,"问题9","window-9")
        self.assertEqual(ninth["background_revision"],1)
        self.assertEqual(ninth["conversation_id"],results[-1]["conversation_id"])
        self.assertEqual(len(self.engine.calls),10)

    async def test_idle_boundary_changes_id_but_old_cached_result_retains_id(self):
        await self.launch()
        reader,peer,_=await self.connect()
        first=await self.ask(reader,peer)
        await self.control(reader,peer,"playback_done","one")
        self.clock.set(10)
        second=await self.ask(reader,peer,"新的对话","two")
        self.assertNotEqual(first["conversation_id"],second["conversation_id"])
        self.assertEqual(self.engine.calls[-1],[{"role":"user","content":"新的对话"}])
        await peer.send({"type":"request_status","id":"one"})
        self.assertEqual(await read_message(reader,1),first)

    async def test_manual_reset_changes_result_id_without_model_metadata(self):
        await self.launch()
        reader,peer,_=await self.connect()
        first=await self.ask(reader,peer)
        await self.control(reader,peer,"playback_done","one")
        self.server.manager.reset("stable-random-session")
        second=await self.ask(reader,peer,"重置后","two")
        self.assertNotEqual(first["conversation_id"],second["conversation_id"])
        self.assertEqual(self.engine.calls[-1],[{"role":"user","content":"重置后"}])
        self.assertEqual(self.server.manager.history("stable-random-session"),[
            {"role":"user","content":"重置后"},{"role":"assistant","content":"回答：重置后"}])

    async def test_heartbeat_is_transport_only_and_does_not_hold_idle(self):
        await self.launch()
        reader,peer,_=await self.connect()
        await self.ask(reader,peer)
        await self.control(reader,peer,"playback_done","one")
        self.clock.set(9)
        await peer.send({"type":"ping"})
        self.assertEqual(await read_message(reader,1),{"type":"pong"})
        self.clock.set(10)
        self.assertEqual(self.server.manager.history("stable-random-session"),[])

    async def test_cache_is_bounded_without_evicting_active_inference(self):
        await self.launch(block_text="慢问题")
        self.server.request_cache.max_completed=2
        slow_r,slow_p,_=await self.connect("slow-session")
        await slow_p.send({"type":"request","id":"slow","text":"慢问题","end":True})
        await read_message(slow_r,1)
        await legacy.until(self.engine.started.is_set)
        # The FIFO model is blocked, so exercise bounded final receipts with
        # model-independent cached errors while the genuine request stays live.
        cache=self.server.request_cache
        for number in range(5):
            receipt=cache.register(f"finished-{number}","id","text")
            cache.finish(receipt,{"type":"error","id":"id","code":"handler_error","message":"failed"})
            cache.expire()
        self.assertEqual(len(cache.receipts),3)
        self.assertIsNone(cache.get("finished-0","id"))
        self.assertIsNotNone(cache.get("slow-session","slow"))
        self.engine.release.set()
        self.assertEqual((await read_message(slow_r,1))["type"],"result")
        self.assertLessEqual(len(cache.receipts),2)

    async def test_close_cancels_all_connection_and_cache_waiters(self):
        await self.launch(block_text="慢问题")
        reader,peer,_=await self.connect()
        await peer.send({"type":"request","id":"pending","text":"慢问题","end":True})
        await read_message(reader,1)
        await legacy.until(self.engine.started.is_set)
        await self.server.close()
        self.assertEqual(self.server.manager.pending_count,0)
        self.assertFalse(self.server.request_cache.receipts)
        self.assertFalse(self.server._completion_tasks)
        self.assertEqual(await asyncio.wait_for(reader.read(),1),b"")


if __name__=="__main__":
    unittest.main()
