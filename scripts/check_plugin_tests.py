import os
import sys

PLUGIN_DIRS = {
    "env": "src/env/",
    "agent": "src/agents/",
    "plugin": "src/plugins/",
}
TEST_DIRS = {
    "env": "tests/environments/",
    "agent": "tests/agents/",
    "plugin": "tests/plugins/",
}


def plugin_name_from_file(fname):
    return os.path.splitext(os.path.basename(fname))[0].lower()


def check_plugin_tests():
    missing = []
    for ptype, pdir in PLUGIN_DIRS.items():
        tdir = TEST_DIRS[ptype]
        if not os.path.exists(pdir):
            continue
        for fname in os.listdir(pdir):
            if not fname.endswith(".py") or fname.startswith("__"):
                continue
            pname = plugin_name_from_file(fname)
            # Look for test file
            expected_test = f"test_{pname}.py"
            test_path = os.path.join(tdir, expected_test)
            if not os.path.exists(test_path):
                missing.append((ptype, fname, test_path))
    if missing:
        print("Missing tests for the following plugins:")
        for ptype, fname, test_path in missing:
            print(f"  [{ptype}] {fname} -> {test_path}")
        sys.exit(1)
    print("All plugins have corresponding tests!")
    sys.exit(0)


if __name__ == "__main__":
    check_plugin_tests()
