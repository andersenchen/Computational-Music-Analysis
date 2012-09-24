import smtplib
import string
import time
import random
import math
import sys
import re

SUBJECT = "PLEASE CONTACT:"
FROM = "shakespeare@shakespeare.info"

phoneNumbers = [];
input = open('numbers.txt', 'r')
p = re.compile('[0-9]');
myNumbers = "";
for line in input:
    if len(line.strip()) > 0:
        phoneNumbers.append((myNumbers.join(p.findall(line))))
    ##print line;
file = open('test.txt', 'r')
suffixes = ["@messaging.sprintpcs.com", "@vtext.com", "@txt.att.net", "@tmomail.net", "@vmobl.com", "@email.uscc.net", "@messaging.nextel.com", "@myboostmobile.com", "@message.alltel.com"]

server = smtplib.SMTP("mail-relay.brown.edu")
startTime = time.time()
endHour = 60*60*10;

for line in file.split("."):
    if len(line.strip()) < 1:
        continue;
    FROM = "god" + str(random.randint(0,10000))+"@heaven.info"
    if time.time()-startTime > endHour:
        print "Program has run!"
        sys.exit(0)
    print line
    for entry in suffixes:
        for phoneNumber in phoneNumbers:
            text = line
            TO = str(phoneNumber)+str(entry);
            BODY = string.join((
                "From: %s" % FROM,
                "To: %s" % TO,
                "Subject: %s" % SUBJECT ,
                "",
                text
                ), "\r\n")
            
            print TO
            server.sendmail(FROM, TO, BODY)
    t = math.fabs(random.gauss(15,5))
    print t
    time.sleep(t)
server.quit()

