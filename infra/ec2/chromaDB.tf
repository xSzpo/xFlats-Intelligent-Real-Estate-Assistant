resource "aws_key_pair" "chroma_key" {
  key_name   = "chroma_key"
  public_key = file("~/.ssh/id_rsa.pub")
}

resource "aws_security_group" "chroma_sg" {
  name        = "chroma_sg"
  description = "Allow SSH and HTTP access"

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "ChromaDB HTTP"
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_instance" "chroma_instance" {
  ami                  = "ami-03250b0e01c28d196" // Ubuntu 24.04 LTS
  instance_type        = "t3.micro"
  key_name             = aws_key_pair.chroma_key.key_name
  security_groups      = [aws_security_group.chroma_sg.name]
  iam_instance_profile = data.terraform_remote_state.iam.outputs.ec2_chroma_instance_profile.name

  root_block_device {
    volume_size = 8
    volume_type = "gp3"
    tags = {
      Name = "chroma-root-volume"
    }
  }

  user_data = file("user_data.sh")
  tags = {
    Name = "ChromaDB-Instance"
  }
}

resource "aws_ebs_volume" "chroma_data" {
  availability_zone = aws_instance.chroma_instance.availability_zone
  size              = 8
  type              = "gp3"

  tags = {
    Name = "chroma-index-data"
  }
}

resource "aws_volume_attachment" "chroma_data_attach" {
  device_name  = "/dev/xvdf"
  volume_id    = aws_ebs_volume.chroma_data.id
  instance_id  = aws_instance.chroma_instance.id
  force_detach = true
}

resource "aws_eip" "chroma_ip" {
  instance   = aws_instance.chroma_instance.id
  domain     = "vpc"
  depends_on = [aws_instance.chroma_instance]
}

output "chroma_public_ip" {
  description = "Elastic IP address of the ChromaDB EC2 instance"
  value       = aws_eip.chroma_ip.public_ip
}

resource "aws_backup_vault" "chroma_backup_vault" {
  name = "chroma-backup-vault"
}

resource "aws_backup_plan" "chroma_backup_plan" {
  name = "chroma-backup-plan"

  rule {
    rule_name         = "weekly-backup"
    target_vault_name = aws_backup_vault.chroma_backup_vault.name
    schedule          = "cron(0 0 ? * 1 *)"
    start_window      = 60
    completion_window = 180

    lifecycle {
      delete_after = 30
    }
  }
}

resource "aws_iam_role" "backup_role" {
  name = "chroma-backup-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Action = "sts:AssumeRole",
        Effect = "Allow",
        Principal = {
          Service = "backup.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "backup_role_attachment" {
  role       = aws_iam_role.backup_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSBackupServiceRolePolicyForBackup"
}

resource "aws_backup_selection" "chroma_backup_selection" {
  name         = "chroma-ebs-selection"
  iam_role_arn = aws_iam_role.backup_role.arn
  plan_id      = aws_backup_plan.chroma_backup_plan.id

  resources = [
    aws_ebs_volume.chroma_data.arn
  ]
}