#! python3.7

import argparse
import asyncio
import io
import os
import speech_recognition as sr
import websockets
import whisper
import torch

from datetime import datetime, timedelta
from queue import Queue
from tempfile import NamedTemporaryFile
from time import sleep
from sys import platform

CHAT_WAKE_UP_PHRASE = ["mike", "mic", "mikie", "michael", "mikee", "miky", "mik", "miky", "mikey"]
# Todo accepts any phrase that starts with "Hey Mike" or "hey mark" or "hey micheal" "hey, mike," "hemai" "hemi" and so on
TERMINAL_WAKE_UP_PHRASE = "Hey Terminal"
async def run_audio_transcription():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="small", help="Model to use",
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--non_english", action='store_true',
                        help="Don't use the english model.")
    parser.add_argument("--energy_threshold", default=1000,
                        help="Energy level for mic to detect.", type=int)
    parser.add_argument("--record_timeout", default=2,
                        help="How real time the recording is in seconds.", type=float)
    parser.add_argument("--phrase_timeout", default=3,
                        help="How much empty space between recordings before we "
                             "consider it a new line in the transcription.", type=float)
    if 'linux' in platform:
        parser.add_argument("--default_microphone", default='pulse',
                            help="Default microphone name for SpeechRecognition. "
                                 "Run this with 'list' to view available Microphones.", type=str)
    args = parser.parse_args()

    # The last time a recording was retreived from the queue.
    phrase_time = None
    # Current raw audio bytes.
    last_sample = bytes()
    # Thread safe Queue for passing data from the threaded recording callback.
    data_queue = Queue()
    # We use SpeechRecognizer to record our audio because it has a nice feauture where it can detect when speech ends.
    recorder = sr.Recognizer()
    recorder.energy_threshold = args.energy_threshold
    # Definitely do this, dynamic energy compensation lowers the energy threshold dramtically to a point where the SpeechRecognizer never stops recording.
    recorder.dynamic_energy_threshold = False

    # Important for linux users. 
    # Prevents permanent application hang and crash by using the wrong Microphone
    if 'linux' in platform:
        mic_name = args.default_microphone
        if not mic_name or mic_name == 'list':
            print("Available microphone devices are: ")
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                print(f"Microphone with name \"{name}\" found")
            return
        else:
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                if mic_name in name:
                    source = sr.Microphone(sample_rate=16000, device_index=index)
                    break
    else:
        source = sr.Microphone(sample_rate=16000)

    # Load / Download model
    model = args.model
    if args.model != "large" and not args.non_english:
        model = model + ".en"
    audio_model = whisper.load_model(model)

    record_timeout = args.record_timeout
    phrase_timeout = args.phrase_timeout

    temp_file = NamedTemporaryFile().name
    transcription = ['']

    with source:
        recorder.adjust_for_ambient_noise(source)

    def record_callback(_, audio: sr.AudioData) -> None:
        """
        Threaded callback function to recieve audio data when recordings finish.
        audio: An AudioData containing the recorded bytes.
        """
        # Grab the raw bytes and push it into the thread safe queue.
        data = audio.get_raw_data()
        data_queue.put(data)

    # Create a background thread that will pass us raw audio bytes.
    # We could do this manually but SpeechRecognizer provides a nice helper.
    recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

    # Cue the user that we're ready to go.
    print("Model loaded.\n")

    while True:
        try:
            # print("Audio transcription running...")
            now = datetime.utcnow()
            # Pull raw recorded audio from the queue.
            if not data_queue.empty():
                phrase_complete = False
                # If enough time has passed between recordings, consider the phrase complete.
                # Clear the current working audio buffer to start over with the new data.
                if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                    last_sample = bytes()
                    phrase_complete = True
                # This is the last time we received new audio data from the queue.
                phrase_time = now

                # Concatenate our current audio data with the latest audio data.
                while not data_queue.empty():
                    data = data_queue.get()
                    last_sample += data

                # Use AudioData to convert the raw data to wav data.
                audio_data = sr.AudioData(last_sample, source.SAMPLE_RATE, source.SAMPLE_WIDTH)
                wav_data = io.BytesIO(audio_data.get_wav_data())

                # Write wav data to the temporary file as bytes.
                with open(temp_file, 'w+b') as f:
                    f.write(wav_data.read())

                # Read the transcription.
                result = audio_model.transcribe(temp_file, fp16=torch.cuda.is_available())
                text = result['text'].strip()

                # If we detected a pause between recordings, add a new item to our transcripion.
                # Otherwise edit the existing one.
                if phrase_complete:
                    transcription.append(text)
                else:
                    transcription[-1] = text

                # Clear the console to reprint the updated transcription.
                os.system('cls' if os.name == 'nt' else 'clear')
                # await run_audio_transcription_queue.put(transcription)
                for line in transcription:
                    print(line)
                    await run_audio_transcription_queue.put(line)
                    print('Audio transcription data added to queue')
                    # words = line.lower().split()
                    # if words and ((any(word in CHAT_WAKE_UP_PHRASE for word in words) or words[-1].rstrip(
                    #         '?!.') in CHAT_WAKE_UP_PHRASE)):
                    #     print('wakeup phrase found')
                    #     await run_audio_transcription_queue.put(line)
                    #     print('Audio transcription data added to queue')
                # Flush stdout.
                print('', end='', flush=True)

                # Infinite loops are bad for processors, must sleep.
                sleep(0.25)
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            break

    print("\n\nTranscription:")
    for line in transcription:
        print(line)


# async def websocket_handler(websocket, path):
#     print("Websocket connected")
#     while True:
#         message = await websocket.recv()
#         print(f"Received message: {message}")
#         response = f"Server received: {message}"
#         await websocket.send(response)
#     pass


# start_server = websockets.serve(websocket_handler, "localhost", 8766)
#
# asyncio.get_event_loop().run_until_complete(start_server)
# asyncio.get_event_loop().run_forever()


async def websocket_handler(websocket, path):
    # handle incoming websocket connections
    await websocket.send("Hello, client!")
    try:
        while True:
            try:
                # receive data from the audio transcription coroutine and send it to all connected clients
                data = await asyncio.wait_for(run_audio_transcription_queue.get(), timeout=1)
                print(f"Received data from audio transcription coroutine: {data}")
                # send data to all connected clients
                await asyncio.wait([websocket.send(data)])
            except asyncio.TimeoutError:
                # continue the loop if no data is received within the timeout period
                continue
    except websockets.exceptions.ConnectionClosed:
        pass

async def run_websockets_server():
    async with websockets.serve(websocket_handler, 'localhost', 8766):
        print("WebSocket server running at ws://localhost:8766")
        await asyncio.Future()  # wait forever


async def main():
    global run_audio_transcription_queue
    run_audio_transcription_queue = asyncio.Queue()
    # create tasks for the websockets server and the audio transcription
    tasks = [
        asyncio.create_task(run_websockets_server()),
        asyncio.create_task(run_audio_transcription())
    ]

    # wait for all tasks to complete (this will never happen since the tasks run indefinitely)
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
