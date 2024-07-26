

echo "$ACCESS_KEY_ID:$ORACLE_CLOUD_SECRET" > passwd && chmod 600 passwd
URL=https://lrtfqsmony6u.compat.objectstorage.uk-london-1.oraclecloud.com 
s3fs -f -d $OCI_BUCKET $MOUNT_POINT -o passwd_file=passwd -o url=$URL -o nomultipart -o use_path_request_style

# to run the container with this scrip, use the following command

# docker run --privileged -d --name s3fs-test -e ORACLE_CLOUD_SECRET=secret -e OCI_BUKCET=bio-gpt-embedding  -e REGION=uk-london-1 -e TENANT_ID=tenant s3fs
