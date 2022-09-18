#! /usr/bin/env python

import cmd
import getpass
import logging
import os
import os.path
import sys

import ConfigParser

from optparse import OptionParser

from nettools.provider.AwsProvider import AwsProvider
from nettools.utils.ipv4_utils import *

logging.basicConfig(format='%(levelname)s %(message)s', level=logging.INFO)
#logging.basicConfig(format='%(levelname)s %(message)s', level=logging.DEBUG)
logger = logging.getLogger()

resources = {}

def checkEnvironments():
    # AWS env
    (tf, aws_credential) = checkAwsEnvironment()
    if tf == False:
        # Add AWS credential manually
        logger.info("AWS Credential does not exist")
        yn = raw_input("Add AWS Credential (y/n)?")
        if yn.lower() == "y":
            aki = raw_input("AWS Access Key ID: ")
            sak = raw_input("AWS Secret Access Key: ")
            aws_credential = (aki, sak)
            tf = True

    if tf == True:
        aws = AwsProvider()
        aws.verifyCredential(aws_credential)
        resources['aws'] = aws
    # OpenStack env
    # TODO

def checkAwsEnvironment():
    """
    Check AWS credential
    """
    user = getpass.getuser()
    userhome = os.path.expanduser('~')
    logger.debug("User: %s" % user)
    logger.debug("Home Directory: %s" % userhome)

    # Find Default AWS credentials
    # ~/.aws/credentials
    cpath1 = "%s/.aws/credentials" % userhome
    cpath2 = "./crednetials"
    cpath = None
    success = False
    if os.path.isfile(cpath1) == True:
        logger.info("Found User's AWS credentials file at %s" % cpath1)
        cpath = cpath1
    elif os.path.isfile(cpath2) == True:
        logger.info("Found User's AWS credentials file at %s" % cpath2)
        cpath = cpath2
    else:
        logger.warn("There is no credentials")
        credential = None
    if cpath:
        # Read AWS credential
        logger.info("Loading %s" % cpath)
        return AwsProvider().loadCredential(cpath)

    return (success, credential)
       
class NetToolsShell(cmd.Cmd):
    intro = """
###################################################################
Welcome to the net-tools shell.
Discover resources first, (for AWS, type "discover aws")
Type help or ?.
###################################################################
"""
    prompt = 'net-tools> '
    file = None

    def do_exit(self, line):
        sys.exit()

    def do_quit(self, line):
        sys.exit()

    def do_discover(self, line):
        """discover [provider] [region_name]
        Discover Provider's resources
        ex) discover aws
        ex) discover aws us-west-1
        ex) discover aws us-west-1 us-east-1"""
        (cmd, params) = self.parseCmd(line)
        if cmd == "aws" and resources.has_key("aws"):
            if params:
                resources["aws"].discover(params)
            else:
                resources["aws"].discover()

    def do_list(self, line):
        """list <cmd> [param1] [param2] ...
        List resources
        ex) list vpc
        ex) list vpc us-west-1
        ex) list vpc us-west-1 us-east-1"""
        (cmd, params) = self.parseCmd(line)
        if cmd == False:
            logger.error("Wrong arguments")
            logger.error("cmd: list <cmd> [param1] [param2] ...")
            return
        # Check Resources
        if len(resources.keys()) == 0:
            logger.warn("There is no provider")
            logger.warn("Discover provider first (type: help discover)")
            return
        logger.debug("Found resources:%s" % resources.keys()) 
        result = []
        if cmd == "vpc":
            self.printHeader("vpc")
            for R in resources.keys():
                result = resources[R].formatVpcs(region_names = params)
                self.printFormat("vpc",R, result)

        else:
            logger.error("Wrong arguments")
            logger.error("cmd: list <cmd> [param1] [param2] ...")
            return

    def do_routable(self, line):
        """routable <VpcId|SubnetId|IP Prefix> <VpcId|SubnetId|IP Prefix>
        Compare IP range conflict between two items
        ex) routable vpcid-00001 vpcid-00002
        ex) routable vpcid-00002 subnetid-00001
        ex) routable 192.168.0.0/24 vpcid-00003
        ex) routable 10.0.0.0/24    10.0.0.128/25"""
        items = line.split()
        if len(items) != 2:
            logger.error("Wrong format")
            logger.error("cmd: routable  <VpcId|SubnetId|IP Prefix> <VpcId|SubnetId|IP Prefix>")
            return
        # Find from VPC
        (a,b) = (items[0], items[1])
        prefix_a = a
        prefix_b = b
        for res in resources.keys():
            R = resources[res]
            if R.getCidrByVpcId(a):
                prefix_a = R.getCidrByVpcId(a)
                break
            elif R.getCidrBySubnetId(a):
                prefix_a = R.getCidrBySubnetId(a)
                break
        for res in resources.keys():
            R = resources[res]
            if R.getCidrByVpcId(b):
                prefix_b = R.getCidrByVpcId(b)
                break
            elif R.getCidrBySubnetId(b):
                prefix_b = R.getCidrBySubnetId(b)
                break
        logger.debug("Network A:%s" % prefix_a)
        logger.debug("Network B:%s" % prefix_b)
        (tf, ips) = overlap(prefix_a, prefix_b)
        (a1,a2) = getNB(prefix_a)
        (b1,b2) = getNB(prefix_b)
        print "Network:%-15s %-15s ~ %-15s" % (a, a1, a2)
        print "Network:%-15s %-15s ~ %-15s" % (b, b1, b2)
        print "-------------------------------------------------"
        if tf == False:
            print "Overlap IP ranges: None"
        else:
            print "Overlap IP ranges: %s ~ %s" % ips

    def parseCmd(self, args):
        items = args.split()
        if len(items) == 0:
            return (False,None)
        cmd = items[0]
        logger.debug("Command:%s" % cmd)
        params = None
        if len(items) > 1:
            params = items[1:]
        return (cmd, params)

    def printHeader(self, o_type):
        """
        @param o_type: output type
        """
        # 'us-west-1', 'vpc-34c5ab51', '-', '172.31.0.0/16', 'us-west-1c', 'subnet-5b26b802', '-', '172.31.0.0/20'
        if o_type == "vpc":
            msg = "%-10s %-15s %-15s %-15s %-15s %-15s %-15s %-15s %-15s" % ("Provider","Region","VpcId","VpcName","CIDR","AZ", "SubnetId","SubnetName","CIDR")
        print msg
        print "-" * 140

    def printFormat(self, o_type, p_type, o_list):
        """
        @param o_type: output type (vpc|ec2|subnet ...)
        @param p_type: provider type (aws|openstack ...)
        @param o_list: output list
        """
        for I in o_list:
            msg = "%-10s %-15s %-15s %-15s %-15s %-15s %-15s %-15s %-15s" % (p_type, I[0], I[1], I[2], I[3], I[4], I[5], I[6], I[7])
            print msg

def main():
    # check credentials (aws, openstack ...)
    checkEnvironments()

    NetToolsShell().cmdloop()

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-L","--logging",dest="logging", help="logging level(DEBUG|INFO|WARNING|ERROR|CRITICAL)")

    try:
        (options,args) = parser.parse_args()
        if options.logging:
            LEVEL = {}
            LEVEL['DEBUG'] = logging.DEBUG
            LEVEL['WARNING'] = logging.WARNING
            LEVEL['INFO'] = logging.INFO
            LEVEL['ERROR'] = logging.ERROR
            LEVEL['CRITICAL'] = logging.CRITICAL

            logging.basicConfig(format='%(levelname)s %(message)s', level=LEVEL[options.logging])
            logger = logging.getLogger('net-tools')
        else:
            logging.basicConfig(format='%(levelname)s %(message)s', level=logging.INFO)
            logger = logging.getLogger('net-tools')
    except:
        parser.print_help()
        sys.exit()

    main()
