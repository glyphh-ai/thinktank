"""step1 — calibrate the ternary bank. Gate T0: >= 256 certified axes, and
both v08 meter axes (good-bad, someone-something) must be certified — the
whole point of the dead zone is that stable extremes + neutral abstention
rescues meters that v09's all-words sign-stability rejected.
"""
from ternary import best_layer, calibrate, save_bank

T0_MIN_AXES = 256
REQUIRED = ("good-bad", "someone-something")


def main():
    layer, rel = best_layer()
    bank = calibrate(layer)
    save_bank(bank)
    names = list(bank["names"])
    n = len(names)
    print(f"layer {layer} (reliability {rel:.3f}): {n}/1830 axes certified "
          f"(agree>=0.95, coverage>=0.10)")
    print(f"median tau {float(sorted(bank['tau'])[n // 2]):.2f}, "
          f"median coverage {float(sorted(bank['coverage'])[n // 2]):.2f}")
    for r in REQUIRED:
        if r in names:
            i = names.index(r)
            print(f"  {r}: certified, tau={bank['tau'][i]:.2f}, "
                  f"coverage={bank['coverage'][i]:.2f}")
        else:
            print(f"  {r}: NOT certified")
    ok = n >= T0_MIN_AXES and all(r in names for r in REQUIRED)
    print(f"T0 {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    main()
