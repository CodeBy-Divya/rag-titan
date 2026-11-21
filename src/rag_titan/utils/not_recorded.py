"""
The error a replay raises when it has no recording for what it was asked.

The playground catches it to say "not recorded for these settings" and to show
how far the run got, instead of failing. It is a ValueError, so code written
for the older plain ValueError keeps working.
"""


class NotRecorded(ValueError):
    """A replayed model or embedder was asked something that was never recorded."""
