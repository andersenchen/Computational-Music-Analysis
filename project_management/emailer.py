#!/usr/bin/python
import smtplib
import string
import time
import random
import math
import sys
import re
import argparse
import textwrap

SUBJECT = ''
spammers = ["messaging.sprintpcs.com", "vtext.com", "txt.att.net", "tmomail.net", "vmobl.com", "email.uscc.net", "messaging.nextel.com", "myboostmobile.com", "message.alltel.com"]
server = smtplib.SMTP("mail-relay.brown.edu")
startTime = time.time()
endHour = 60*60*10

ap = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description=textwrap.dedent('''\
     project management software to spam employee with texts
     e.g. %s CUM TO 3RD FLOOR -to 6176866118
     ''' % __file__)
    )
ap.add_argument('message', nargs='+', help='what to spam')
ap.add_argument('-to', dest='phonenumbers', nargs='+', help='who to spam')
args = ap.parse_args()
message = ' '.join(args.message)
phonenumbers = args.phonenumbers
print 'message:     ', message
print 'phonenumbers:', phonenumbers

for phonenumber in phonenumbers:
    FROM = "god" + str(random.randint(0,10000))+"heaven.info"
    if time.time()-startTime > endHour:
        server.quit()
        sys.exit(0)
    print phonenumber
    
    for spammer in spammers:
        TO = phonenumber + '@' + spammer
        BODY = string.join((
                "From: %s"    % FROM,
                "To: %s"      % TO,
                "Subject: %s" % SUBJECT ,
                "",
                message
                ), "\r\n")
        
        print TO
        server.sendmail(FROM, TO, BODY)

    t = math.fabs(random.gauss(15,5))
    print t
    time.sleep(t)

