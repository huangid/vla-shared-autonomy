"""Print raw SpaceMouse HID reports. No Isaac, no policy — just the device.

Use when the arm responds but the gripper does not (or vice versa): it separates
"the device never sent it" from "the sim ignored it".

    python scripts/check_spacemouse.py

Move the puck and press both buttons. Expect:
  report 1 = translation, report 2 = rotation, report 3 = buttons.
`spacemouse_pilot.SpaceMousePilot` maps report 3 bit 0 -> close, bit 1 -> open; if
your buttons arrive under a different report id or different bits, that mapping is
what needs changing.
"""
import sys
import time

import hid

VENDOR, PRODUCT = 0x256F, 0xC635     # 3Dconnexion SpaceMouse Compact


def main():
    dev = hid.device()
    try:
        dev.open(VENDOR, PRODUCT)
    except OSError as exc:
        raise SystemExit(
            f"cannot open {VENDOR:#06x}:{PRODUCT:#06x} ({exc}).\n"
            "  - plugged in?            lsusb -d 256f:\n"
            "  - already held by a run? pgrep -af 'play.py|shared_autonomy.py'\n"
            "  - permissions?           ls -l /dev/hidraw*  (needs crw-rw-rw-)"
        )
    dev.set_nonblocking(True)
    print(f"opened: {dev.get_product_string()}")
    print("move the puck and press BOTH buttons; Ctrl-C to stop\n")

    seen_reports = set()
    seen_buttons = set()
    t0 = time.time()
    try:
        while True:
            data = dev.read(64)
            if not data:
                time.sleep(0.002)
                continue
            rid = data[0]
            if rid not in seen_reports:
                seen_reports.add(rid)
                print(f"[new] report id {rid}: {data[:8]}")
            if rid == 3:
                mask = data[1] if len(data) > 1 else 0
                if mask and mask not in seen_buttons:
                    seen_buttons.add(mask)
                print(f"\rbuttons: mask={mask:#04x}  "
                      f"bit0(close)={'Y' if mask & 0x01 else 'n'}  "
                      f"bit1(open)={'Y' if mask & 0x02 else 'n'}   ", end="", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        dev.close()

    print("\n\n--- summary ---")
    print(f"ran {time.time() - t0:.0f}s | report ids seen: {sorted(seen_reports)}")
    if 3 in seen_reports:
        print(f"button masks seen: {[hex(m) for m in sorted(seen_buttons)] or 'NONE — buttons never registered'}")
        if seen_buttons and not (seen_buttons & {0x01, 0x02}):
            print("Buttons arrive on report 3 but NOT as bit 0 / bit 1 — spacemouse_pilot.py's\n"
                  "mapping does not match this device. Use the masks above to fix it.")
    else:
        print("No report id 3 seen: this device does not send buttons that way.\n"
              "Check the other report ids above for a byte that changes when you press.")
    sys.exit(0)


if __name__ == "__main__":
    main()
