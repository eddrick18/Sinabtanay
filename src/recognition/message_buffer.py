"""Explicit keyboard editing of a bounded, in-memory fingerspelling message."""
class MessageBuffer:
    def __init__(self, labels, limit=200):
        self.labels=set(labels)
        self.limit=limit
        self.text=''

    def handle_key(self, key, stable_label):
        if key in (10,13):
            if stable_label not in self.labels or len(str(stable_label))!=1:
                return 'Wait for a stable letter before adding'
            if len(self.text)>=self.limit:
                return 'Message full - delete or clear'
            self.text+=stable_label
            return f'Added {stable_label}'
        if key==32:
            if not self.text or self.text.endswith(' '):
                return 'Space skipped'
            if len(self.text)>=self.limit:
                return 'Message full - delete or clear'
            self.text+=' '
            return 'Added space'
        if key in (8,127):
            self.text=self.text[:-1]
            return 'Deleted last character'
        if key in (ord('c'),ord('C')):
            self.text=''
            return 'Message cleared'
        return None
