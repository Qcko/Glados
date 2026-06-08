"""GLaDOS room client — a lite (thin, headless) transducer client.

A room client is a dumb single-transducer endpoint (ARCH §2): a mic client
streams audio up, a speaker client plays TTS down. No UI, no per-turn state —
the server's Organizer owns rooms/sessions/routing. Built to run on the dev
box (for development) and on an old phone running Android + Termux (the intended
appliance), so it stays pure portable Python and never imports the server
package (`src/glados`): the shared wire contract is vendored in `wire.py`.

Ships the mic client (`client_room.mic`), the speaker client
(`client_room.speaker`), and the room supervisor (`client_room.room`) that runs
both in one process.
"""
