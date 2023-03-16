# take mention part of utterance from mention detection module
# model_ambiguity to find potential candidate entities
#   if more than one candidate:
#       compute mention-entity probability
#       compute entity priors
#       compute cost of mention
#       perform RSA computation
#       if above threshold:
#           assign ref id to mention and add label
#           add position and new features/labels to brain
#       if below threshold:
#           detect missing information (?)
#           phrase clarification question
#           ask clarification question

