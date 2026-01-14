aws secretsmanager restore-secret --secret-id chrome-db-011337673661


Check the log of your user data script in:
tail -n 30 /var/log/cloud-init.log
tail -n 30 /var/log/cloud-init-output.log