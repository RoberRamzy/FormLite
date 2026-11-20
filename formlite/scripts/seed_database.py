"""Create the Room prepackaged database from the reference SQLite schema."""

import argparse
import re
import sqlite3
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('schema', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    sql = args.schema.read_text(encoding='utf-8')
    sql = re.sub(r'CREATE VIEW (\w+) AS\s+(.*?);',
                 lambda m: f'CREATE VIEW `{m[1]}` AS {" ".join(m[2].split())};', sql, flags=re.S)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    staged = args.output.with_suffix('.tmp')
    staged.unlink(missing_ok=True)
    with sqlite3.connect(staged) as connection:
        connection.executescript(sql)
        connection.execute('PRAGMA user_version=1')
        assert not connection.execute('PRAGMA foreign_key_check').fetchall()
    staged.replace(args.output)


if __name__ == '__main__':
    main()
