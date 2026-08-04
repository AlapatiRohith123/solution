from __future__ import annotations

import pass_fail
import rubric_score


def main() -> int:
    rubric_score.main()
    return pass_fail.main()


if __name__ == "__main__":
    raise SystemExit(main())
