import pandas as pd
from datasets import Dataset

def create_comprehensive_yoda_dataset():
    """Create extensive Yoda training data with various sentence types"""
    
    data = [
        # Basic statements
        {"input": "You are very strong.", "target": "Strong you are, very strong."},
        {"input": "I am your teacher.", "target": "Your teacher, I am."},
        {"input": "The Force is powerful.", "target": "Powerful the Force is."},
        {"input": "This task is difficult.", "target": "Difficult this task is."},
        {"input": "You have much potential.", "target": "Much potential you have."},
        
        # Action statements
        {"input": "I will train you well.", "target": "Train you well, I will."},
        {"input": "You must learn patience.", "target": "Patience you must learn, hmm."},
        {"input": "We should trust the Force.", "target": "Trust the Force, we should."},
        {"input": "You can become a Jedi.", "target": "A Jedi you can become."},
        {"input": "I need to meditate now.", "target": "Meditate now, I must."},
        
        # Emotional statements
        {"input": "Fear leads to anger.", "target": "To anger, fear leads."},
        {"input": "Anger leads to hate.", "target": "To hate, anger leads."},
        {"input": "Hate leads to suffering.", "target": "To suffering, hate leads."},
        {"input": "I sense great fear in you.", "target": "Great fear in you, I sense."},
        {"input": "Your feelings betray you.", "target": "Betray you, your feelings do."},
        
        # Questions and advice
        {"input": "Do you understand?", "target": "Understand, do you?"},
        {"input": "What do you see?", "target": "See, what do you?"},
        {"input": "Trust your instincts.", "target": "Your instincts, trust you must."},
        {"input": "Control your emotions.", "target": "Control your emotions, you must."},
        {"input": "Search your feelings.", "target": "Search your feelings, you should."},
        
        # Force-related wisdom
        {"input": "The Force flows through everything.", "target": "Through everything, the Force flows."},
        {"input": "Size matters not.", "target": "Size matters not. Judge me by my size, do you?"},
        {"input": "There is no try, only do.", "target": "Do or do not. There is no try."},
        {"input": "The dark side is dangerous.", "target": "Dangerous the dark side is."},
        {"input": "Balance brings peace.", "target": "Peace, balance brings."},
        
        # Future tense
        {"input": "You will become wise.", "target": "Wise you will become."},
        {"input": "I shall teach you.", "target": "Teach you, I shall."},
        {"input": "We will face many trials.", "target": "Many trials we will face."},
        {"input": "The truth will reveal itself.", "target": "Reveal itself, the truth will."},
        {"input": "Time will tell.", "target": "Tell us, time will."},
        
        # Past tense
        {"input": "I have seen many things.", "target": "Many things I have seen."},
        {"input": "You were chosen for this.", "target": "Chosen for this, you were."},
        {"input": "The Empire has fallen.", "target": "Fallen, the Empire has."},
        {"input": "I trained many Padawans.", "target": "Many Padawans I trained."},
        {"input": "We fought in the wars.", "target": "In the wars, fought we did."},
        
        # Complex philosophical statements
        {"input": "Knowledge and wisdom are different things.", "target": "Different things, knowledge and wisdom are."},
        {"input": "A Jedi must have the deepest commitment.", "target": "The deepest commitment, a Jedi must have."},
        {"input": "Adventure and excitement, a Jedi craves not these.", "target": "Adventure, excitement... A Jedi craves not these things."},
        {"input": "You must unlearn what you have learned.", "target": "Unlearn what you have learned, you must."},
        {"input": "Difficult to see the future is.", "target": "Difficult to see. Always in motion is the future."},
        
        # Everyday situations (testing versatility)
        {"input": "I'm going to the store.", "target": "To the store, going I am."},
        {"input": "The weather is nice today.", "target": "Nice today, the weather is."},
        {"input": "I like this food.", "target": "Like this food, I do."},
        {"input": "Can you help me?", "target": "Help you, can I?"},
        {"input": "Thank you for everything.", "target": "For everything, thank you I do."},
        
        # Negative statements
        {"input": "I don't understand this.", "target": "Understand this, I do not."},
        {"input": "You cannot give up now.", "target": "Give up now, you cannot."},
        {"input": "This is not the way.", "target": "The way, this is not."},
        {"input": "I will not abandon you.", "target": "Abandon you, I will not."},
        {"input": "Fear you should not.", "target": "Fear you should not, hmm."},
    ]
    
    return Dataset.from_pandas(pd.DataFrame(data))

# # Create the dataset
# yoda_dataset = create_comprehensive_yoda_dataset()
# print(f"\n📚 Created Yoda dataset with {len(yoda_dataset)} examples")

# # Show some examples
# print("\n🔸 Sample training pairs:")
# for i in range(5):
#     print(f"Normal: {yoda_dataset[i]['input']}")
#     print(f"Yoda:   {yoda_dataset[i]['target']}")
#     print()