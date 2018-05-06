from conll_reader import DependencyStructure, DependencyEdge, conll_reader
from collections import defaultdict
import copy
import sys

import numpy as np
import keras

from extract_training_data import FeatureExtractor, State

class Parser(object): 

    def __init__(self, extractor, modelfile):
        self.model = keras.models.load_model(modelfile)
        self.extractor = extractor
        
        # The following dictionary from indices to output actions will be useful
        self.output_labels = dict([(index, action) for (action, index) in extractor.output_labels.items()])

    def check_action(self, action, state):
        if len(state.stack) == 0 and action in {"right_arc", "left_arc"}:
            return False
        if len(state.stack) > 0 and len(state.buffer) == 1 and action == "shift":
            return False
        if len(state.stack) > 0 and state.stack[-1] == 0 and action == "left_arc":
            return False
        return True

    def parse_sentence(self, words, pos):
        state = State(range(1,len(words)))
        state.stack.append(0)
        while state.buffer:
            input_vec = self.extractor.get_input_representation(words, pos, state)
            input_vec = input_vec.reshape((1,6))
            possible_actions = self.model.predict(input_vec)[0].tolist()
            sorted_actions = [i[0] for i in sorted(enumerate(possible_actions), \
                                reverse=True, key=lambda x:x[1])]
            i=0
            while not self.check_action(self.output_labels[sorted_actions[i]][0], state):
                i+=1

            action, dep_rel = self.output_labels[sorted_actions[i]]
            if not dep_rel:
                state.shift()
            elif action == "left_arc":
                state.left_arc(dep_rel)
            elif action == "right_arc":
                state.right_arc(dep_rel)

        result = DependencyStructure()
        for p,c,r in state.deps: 
            result.add_deprel(DependencyEdge(c,words[c],pos[c],p, r))
        return result 
        

if __name__ == "__main__":

    WORD_VOCAB_FILE = 'data/words.vocab'
    POS_VOCAB_FILE = 'data/pos.vocab'

    try:
        word_vocab_f = open(WORD_VOCAB_FILE,'r')
        pos_vocab_f = open(POS_VOCAB_FILE,'r') 
    except FileNotFoundError:
        print("Could not find vocabulary files {} and {}".format(WORD_VOCAB_FILE, POS_VOCAB_FILE))
        sys.exit(1) 

    extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)
    parser = Parser(extractor, sys.argv[1])

    with open(sys.argv[2],'r') as in_file: 
        for dtree in conll_reader(in_file):
            words = dtree.words()
            pos = dtree.pos()
            deps = parser.parse_sentence(words, pos)
            print(deps.print_conll())
            print()
        
