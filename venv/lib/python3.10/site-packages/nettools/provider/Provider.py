##########################
# Base Class
#########################

import logging
from types import *

logger = logging.getLogger("net-tools")

class Provider(object):
    def __init__(self):
        self.regions = {}
        self.vpcs = {}
        self.subnets = {}
        self.instances = {}

    def discover(self, region_names=None):
        """
        @return: None
        """

    def listRegions(self, region_name=None):
        """
        @return: dictionary
        """
        return self.regions

    def listAvailabilityZones(self, region_name):
        """
        List AvailabilityZones of Regions
        @return: dictionary
        """
        pass

    def listVpcs(self, region_name):
        """
        List Vpcs of region
        @return: dictionary
        """
        pass

    def formatVpcs(self, region_name = None):
        pass

    #######################
    # Utilities
    #######################
    def getValue(self, dic, name):
        """
        @param dic: dictionary
        @param name : key
        @return: dic[name]
        """
        if type(dic) != DictType:
            logger.error("%s is not Dictionary type" % dic)
            return None
        if dic.has_key(name) == False:
            logger.error("Dic does not have key:%s" % name)
            return None
        return dic[name]


