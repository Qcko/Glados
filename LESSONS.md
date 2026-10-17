# LESSONS - GLaDOS

Curated by topic, not a log. What was learned the hard way, including what did
not work, so the next session does not relearn it.

## Room phones (Termux)

- **The mic comes from Termux:API, not Termux.** The main Termux app declares
  no microphone permission, so its Android permission page has no toggle to
  grant. PulseAudio's `module-sles-source` captures only once **Termux:API**
  (same shared app identity) holds Microphone. Without it capture is silence
  and nothing errors anywhere: the client connects, the server logs no speech.
  Found on the livingroom phone (06-06-2026), forgotten, re-found on the
  kitchen phone (17-10-2026).
- **Play Store Termux is unusable.** It is abandoned and newer phones refuse
  it. Use F-Droid (or GitHub releases) for Termux and both addons.
- **A fresh F-Droid Termux needs `pkg upgrade` before installs.** Its base is
  old; `python-numpy` installed on top failed at import with "cannot locate
  symbol". `deps.sh` now upgrades once before its first install.
- **`sv` reports "unable to change to service directory" right after
  `termux-services` is installed.** The service dir and runsvdir only come up
  in a new Termux session; exit Termux fully and reopen.
- **Termux:Boot needs Unrestricted battery on Samsung.** The first reboot test
  failed until Termux, Termux:Boot and Termux:API were all Unrestricted and off
  the sleeping-apps lists. Opening Termux:Boot only ever shows a readme; that
  is normal.
- **Test autostart without opening Termux.** Opening Termux starts the
  services itself, so a check done after opening it proves nothing about boot.
  Judge by the server-side connection time instead.

## Speaker identity

- **The fallback speaker is a session bucket, not a person.** Every client's
  `default_user` is a real household member, so treating an unidentified voice
  as `default_user` would guess that person by construction. `SpeakerIdentity`
  splits the two: `session_key` keeps today's room continuity, and `person` is
  None unless the voice matched. Consumers (memory) key on `person` + `verified`.
  A guest sentinel session was considered and rejected 17-10-2026: it splits
  the conversation every time a short "yes" can't be identified.
