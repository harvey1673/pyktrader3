import pythoncom
import win32com.client as win32
import psutil
import os
import subprocess
import smtplib
from email.mime.text import MIMEText


def send_email_w_outlook(recepient, subject, body_text = '', attach_files = [], html_text = ''):
    pythoncom.CoInitialize()
    outlook = win32.Dispatch('outlook.application')
    mail = outlook.CreateItem(0)
    mail.To = recepient
    mail.Subject = subject
    if len(body_text) > 0:
        mail.body = body_text
    if len(html_text)> 0:
        mail.HTMLBody = html_text
    for afile in attach_files:
        if type(afile) is list:
            attachment = mail.Attachments.Add(afile[0])
            attachment.PropertyAccessor.SetProperty("http://schemas.microsoft.com/mapi/proptag/0x3712001F", afile[1])
        else:
            mail.Attachments.Add(Source = afile)
    mail.send


def open_outlook():
    try:
        subprocess.call(['C:\Program Files (x86)\Microsoft Office\Office15\Outlook.exe'])
        os.system("C:\Program Files (x86)\Microsoft Office\Office15\Outlook.exe");
    except:
        print("Outlook didn't open successfully")


def send_email_by_outlook(recepient, subject, body_text = '', attach_files = [], html_text = ''):
    for item in psutil.pids():
        try:
            p = psutil.Process(item)
            if p.name() == "OUTLOOK.EXE":
                flag = 1
                break
            else:
                flag = 0
        except:
            continue
    if (flag == 0):
        open_outlook()
    send_email_w_outlook(recepient, subject, body_text, attach_files, html_text)


def send_email_by_smtp(mail_account, to_list, sub, content):
    mail_host = mail_account['host']
    mail_user = mail_account['user']
    mail_pass = mail_account['passwd']
    msg = MIMEText(content)
    msg['Subject'] = sub
    msg['From'] = mail_user
    msg['To'] = ';'.join(to_list)
    try:
        smtp = smtplib.SMTP(mail_host, mail_account['port'])
        # smtp.ehlo()
        smtp.starttls()
        # smtp.ehlo()
        smtp.login(mail_user, mail_pass)
        smtp.sendmail(mail_user, to_list, msg.as_string())
        smtp.close()
        return True
    except Exception as e:
        print("exception when sending e-mail %s" % str(e))
        return False
