# to prevent eg test_transformers.py from erroring because no tests are run
def pytest_sessionfinish(session, exitstatus):
    if exitstatus == 5:
        session.exitstatus = 0  # Any arbitrary custom status you want to return```
