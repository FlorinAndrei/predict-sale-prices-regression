terraform {
  required_providers {
    aws = {
      source = "hashicorp/aws"
    }
  }
}

provider "aws" {
  region = "us-west-2"
}

variable "cluster_instance_count" {
  default = "10"
}

# env vars
variable "control_ip" {}
variable "aws_default_vpc_ips" {}

resource "aws_security_group" "main_sg" {
  name        = "main_sg"
  description = "main security group"

  tags = {
    Name = "main_sg"
  }
}

resource "aws_security_group_rule" "egress_all" {
  type              = "egress"
  from_port         = 0
  to_port           = 0
  protocol          = "-1"
  cidr_blocks       = ["0.0.0.0/0"]
  ipv6_cidr_blocks  = ["::/0"]
  security_group_id = aws_security_group.main_sg.id
}

resource "aws_security_group_rule" "allow_ssh" {
  type              = "ingress"
  from_port         = 22
  to_port           = 22
  protocol          = "tcp"
  cidr_blocks       = [var.control_ip]
  security_group_id = aws_security_group.main_sg.id
}

resource "aws_security_group_rule" "allow_mysql" {
  type              = "ingress"
  from_port         = 3306
  to_port           = 3306
  protocol          = "tcp"
  cidr_blocks       = [var.control_ip, var.aws_default_vpc_ips]
  security_group_id = aws_security_group.main_sg.id
}

resource "aws_instance" "compute_instance" {
  ami                                  = "ami-03f65b8614a860c29"
  instance_type                        = "c5.4xlarge"
  count                                = var.cluster_instance_count
  key_name                             = "amazon-oregon"
  associate_public_ip_address          = true
  instance_initiated_shutdown_behavior = "terminate"
  security_groups                      = ["main_sg"]

  provisioner "local-exec" {
    command = "echo ${self.public_ip} >> ./inventory-cluster"
  }

  tags = {
    Name = "optuna-${format("%02d", count.index)}"
  }
}

resource "aws_instance" "db_instance" {
  ami                                  = "ami-03f65b8614a860c29"
  instance_type                        = "t3.small"
  key_name                             = "amazon-oregon"
  associate_public_ip_address          = true
  instance_initiated_shutdown_behavior = "stop"
  security_groups                      = ["main_sg"]

  provisioner "local-exec" {
    command = "echo ${self.private_ip} > ./inventory-db-private; echo ${self.public_ip} > ./inventory-db"
  }

  tags = {
    Name = "optuna-db"
  }
}
