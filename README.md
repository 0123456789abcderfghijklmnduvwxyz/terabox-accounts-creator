# terabox-accounts-creator
Creates terabox accounts in mass, uses tor so dw about you getting rate limited

just install all the dependecies, install tor, if you are on linux start the tor service and on windows just search for tor.exe in the search bar, and let it open for as long as you run the scripts, run onionmail_accounts.py to get all the mailbox accounts, its a bit faster than terabox_accounts.py, i have created 10000 accounts in aboout 8 hours of runtime, you do not need that much though, i just had it running for an entire day. Then you just run terabox_accounts.py with some of the arguments i gave down below, and then it'll just start creating those accounts. Right now in the moment im writing this i am not sure if the captcha solver works, because you only sometimes get captcha's, and i for some reason haven't gotten one yet.

Here are some example arguments you can run the script with, here first for onionmail_accounts.py: 

python onionmail_accounts.py --mode tor --browser=chrome --headless=true --stealth=true --accounts=50 --threads=5

And here for terabox_accounts.py, its basically just the same thing with diffrent values for accounts and threads: 

python terabox_accounts.py --mode=tor --browser=chrome --headless=false --stealth=true --accounts=10 --concurrency=2

It is kinda fun to watch terabox_accounts.py creating accounts, onionmail_accounts.py is more boring on that note, but you can run both of them headless and everything works find, and i would even recommend doing that.
