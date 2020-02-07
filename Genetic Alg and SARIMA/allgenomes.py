""" Class that keeps track of all genomes trained so far, and their scores.
    Among other things, ensures that genomes are unique.
"""

import random
import logging

#from genome import Genome
from network import Network

class AllNetworks():
    """Store all genomes
    """

    def __init__(self, firstgenome):
        """Initialize
        """

        self.population = []
        self.population.append(firstgenome)
        
    def add_network(self, network):
        """Add the network to our population.
        """

        for i in range(0,len(self.population)):
            if (network.hash == self.population[i].hash):
                logging.info("add_network() ERROR: hash clash - duplicate network")
                return False

        self.population.append(network)

        return True
        
    def set_accuracy(self, network):
        """Add the network to our population.
        """
        
        for i in range(0,len(self.population)):
            if (network.hash == self.population[i].hash):
                self.population[i].accuracy = network.accuracy
                return
    
        logging.info("set_accuracy() ERROR: network not found")

    def is_duplicate(self, network):
        """Add the network to our population.
        """

        for i in range(0,len(self.population)):
            if (network.hash == self.population[i].hash):
                return True
    
        return False

    def print_all_networks(self):
        """Print out a genome.
        """

        for network in self.population:
            network.print_genome_ma()