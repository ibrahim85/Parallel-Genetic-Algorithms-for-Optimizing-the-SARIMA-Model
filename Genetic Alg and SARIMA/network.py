"""Class that represents the network to be evolved."""
import random
import logging
import hashlib
import copy

from train import train_and_score

class Network():
    """Represent a network and let us operate on it.

    Currently only works for an MLP.
    """

    def __init__(self, nn_param_choices=None, networkparam = {}, u_ID = 0, mom_ID = 0, dad_ID = 0, gen = 0):
        """Initialize our network.

        Args:
            nn_param_choices (dict): Parameters for the network, includes:
                nb_neurons (list): [64, 128, 256]
                nb_layers (list): [1, 2, 3, 4]
                activation (list): ['relu', 'elu']
                optimizer (list): ['rmsprop', 'adam']
        """
        self.accuracy = 0.
        self.param    = (0, 0, 0)
        self.param_seasonal   = (0, 0, 0,12)
        self.nn_param_choices = nn_param_choices
        self.network = networkparam  # (dic): represents MLP network parameters
        self.u_ID             = u_ID
        self.parents          = [mom_ID, dad_ID]
        self.generation       = gen
        
        #hash only makes sense when we have specified the genes
        if not networkparam:
            self.hash = 0
        else:
            self.update_hash()

    def update_hash(self):
        """
        Refesh each genome's unique hash - needs to run after any genome changes.
        """
        genh = str(self.nn_param_choices['p_values']) + str(self.nn_param_choices['d_values']) + str(self.nn_param_choices['q_values']) +str(self.nn_param_choices['sp_values']) + str(self.nn_param_choices['sd_values']) + str(self.nn_param_choices['sq_values'])
        self.hash = hashlib.md5(genh.encode("UTF-8")).hexdigest()
        self.accuracy = 0.0

    def create_random(self):
        """Create a random network."""
        #print("create_random")
        self.parents = [0,0] 
        for key in self.nn_param_choices:
            self.network[key] = random.choice(self.nn_param_choices[key])
        self.update_hash()

    def create_set(self, network):
        """Set network properties.

        Args:
            network (dict): The network parameters

        """
        self.network = network

    def train(self, dataset, type_ser):
        """Train the network and record the accuracy.

        Args:
            dataset (str): Name of dataset to use.

        """
        if self.accuracy == 0.:
            try:
                self.param, self.param_seasonal, self.accuracy = train_and_score(self.network, dataset,type_ser)
            except:
                self.accuracy = train_and_score(self.network, dataset,type_ser)

    def print_network(self):
        """Print out a network."""
        logging.info(self.network)
        logging.info("Network accuracy: %.2f%%" % (self.accuracy))

    def mutate_one_gene(self):
        """Randomly mutate one gene in the genome.

        Args:
            network (dict): The genome parameters to mutate

        Returns:
            (Genome): A randomly mutated genome object

        """
        # Which gene shall we mutate? Choose one of N possible keys/genes.
        #gene_to_mutate = random.choice( list(self.nn_param_choices.keys()) )

        # And then let's mutate one of the genes.
        # Make sure that this actually creates mutation
        #current_value    = self.nn_param_choices[gene_to_mutate]
        #possible_choices = copy.deepcopy(self.nn_param_choices[gene_to_mutate])
        
        #possible_choices.remove(current_value)
        
        #self.nn_param_choices[gene_to_mutate] = random.choice( possible_choices )
        
        # Choose a random key.
        mutation = random.choice(list(self.nn_param_choices.keys()))

        # Mutate one of the params.
        self.network[mutation] = random.choice(self.nn_param_choices[mutation])
        
        self.update_hash()
    
    def set_generation(self, generation):
        """needed when a genome is passed on from one generation to the next.
        the id stays the same, but the generation is increased"""   

        self.generation = generation
        #logging.info("Setting Generation to %d" % self.generation)

    def set_genes_to(self, nn_param_choices, mom_ID, dad_ID):
        """Set genome properties.
        this is used when breeding kids

        Args:
            genome (dict): The genome parameters
        IMPROVE
        """
        self.parents  = [mom_ID, dad_ID]
        
        self.nn_param_choices = nn_param_choices

        self.update_hash()

    def print_genome(self):
        """Print out a genome."""
        #self.print_geneparam()
        logging.info("Acc: %.2f%%" % (self.accuracy ))
        logging.info("UniID: %d" % self.u_ID)
        logging.info("Mom and Dad: %d %d" % (self.parents[0], self.parents[1]))
        logging.info("Gen: %d" % self.generation)
        logging.info("Hash: %s" % self.hash)

    def print_genome_ma(self):
        """Print out a genome."""
        #self.print_geneparam()
        logging.info("Acc: %.2f%% UniID: %d Mom and Dad: %d %d Gen: %d" % (self.accuracy , self.u_ID, self.parents[0], self.parents[1], self.generation))
        logging.info("Hash: %s" % self.hash)