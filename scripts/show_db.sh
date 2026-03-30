#!/bin/bash
# show_db.sh — Display registered plates and access log from the ANPR SQLite database
python3 -c "
import sqlite3
conn = sqlite3.connect('data/anpr_database.db')
print('=' * 60)
print('       REGISTERED PLATES DATABASE')
print('=' * 60)
print('  {:<20s} {:<12s} {}'.format('PLATE', 'STATUS', 'OWNER'))
print('  ' + '-' * 50)
for r in conn.execute('SELECT * FROM plates'):
    print('  {:<20s} {:<12s} {}'.format(r[0], r[1], r[2]))
print('')
print('=' * 60)
print('       ACCESS LOG (last 20)')
print('=' * 60)
print('  {:<20s} {:<12s} {:<8s} {:<8s} {}'.format('PLATE','STATUS','MATCH','CONF','TIMESTAMP'))
print('  ' + '-' * 70)
for r in conn.execute('SELECT * FROM access_log ORDER BY id DESC LIMIT 20'):
    print('  {:<20s} {:<12s} {:<8s} {:.2f}     {}'.format(r[1], r[2], r[3], r[4], r[5]))
conn.close()
"
