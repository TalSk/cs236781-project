=Audio Source Seperation using DDSP

==Abstract (10%). Summarize your work. Briefly introduce the problem, the methods and state the key results.

- Audio Source Seperation using DDSP
√

- Audio synthesis is hard as-is, even for monophonic audio. One pledge of the DDSP library is reducing the large parameter size that is mutual to previous works of audio synthesis.
	We would like to examine this by challenging the library in a problem closer to the "real-world" than the one presented in the article.
	One of these is seperating instruments from a polyphonic audio recording.
√

- Using a datset of music originating from 3 monophonic sources; drums, bass and vocals and their combined output, we trained a deterministic autoencoder as presented in the original paper as a baseline for each source, as well as Conv-TasNet to seperate incoming audio to the fundamental frequencies. Combined together they create a full network that is theoretically capable of audio seperation on these exact sources.
√

- State key results.
X


==Intro (25%). Review the papers relevant to your project. Explain the problem domain, existing approaches and the specific contribution of the relevant paper(s). Also detail the drawbacks which you plan to address. If it’s a custom project, explain your specific motivation and goals. Cite any other work as needed.

- General review of DDSP.
√

- Drawbacks of DDSP, highlight those who pointed us in the separation direction.
√

(Basically copy some of the implementation ideas)
- Review of relevant work on the ASS domain.
√

- 

==Methods (25%). If implementing an existing paper, explain the original approach as well as your ideas for modifications, additions or improvements to the algorithm/task/domain etc., as relevant. Otherwise, provide a detailed explanation of your approach. In both cases, explain the empirical and/or theoretical motivation for what you are doing. Finally, describe the data you will be using for evaluation.

- Explain the theoretical and practical model of the DDSP approach to differntiable elements in the network.
√

(Basically copy some of the implementation ideas)
- Explain 2-3 options of available approaches to ASS using DDSP, as we saw fit. (Explain theoretical motiviation)

- Describe the dataset (Explain theoritically why it is required the sources to be monophonic, and it should be the same creator, for now. Explain how empirically it is supposed to be enough, according to the original paper where they used just 13 minutes to train).
	Include explanation of the audio preprocessing process.


==Implementation and experiments (20%). Describe the experiments performed and their configurations, what was compared to what and the evaluation metrics used and why. Explain all implementation details such as model architectures used, data preprocessing/augmentation approaches, loss formulations, training methods and hyperparameter values.

Note: You can use existing code, e.g. in your implementation but specify what you used and which parts you implemented yourself.

- Explain all code relevant to the learning process - this includes abstractically detailed the whole autoencoder model and how Conv-TasNet fits in.

- Explain how the combination of the network works.

- Detail the relevant matrics:
	Loudness L1 distance (for the TasNet)
	F0  L1 distance (for the TasNet)
	The article's mentioned multi-scale spectal loss (for the whole per-instrument reconstruction)
	CREPE's binary cross entroy loss

- Show experiments to check how well the job was done.
	Possible test cases:
		Same instruments, same artist
		Same instruments, different artist
		Extra 1 instrument

==Results (20%). Present all results in an orderly table and include graphs or figures as you see fit. Discuss, analyze and explain your results. Compare to previous works and other approaches for your task.

- Results discussion

- Experiments results in a table.

- Audio clips of results and experiments.