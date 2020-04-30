## Contributing
I am delighted when people want to contribute to LIME. Here are a few things to keep in mind before sending in a pull request:
* We are now using flake8 as a style guide enforcer (I plan on adding eslint for javascript soon). Make sure your code passes the default flake8 execution.
* There must be a really good reason to change the external interfaces - I want to avoid breaking previous code as much as possible.
* If you are adding a new feature, please let me know the use case and the rationale behind how you did it (unless it's obvious)

If you want to contribute but don't know where to start, take a look at the [issues page](https://github.com/marcotcr/lime/issues), or at the list below.

# Roadmap
Here are a few high level features I want to incorporate in LIME. If you want to work incrementally in any of these, feel free to start a branch.

1. Creating meaningful tests that we can run before merging things. Right now I run the example notebooks and the few tests we have.
2. Creating a wrapper that computes explanations for a particular dataset, and suggests instances for the user to look at (similar to what we did in [the paper](http://arxiv.org/abs/1602.04938))
3. Making LIME work with images in a reasonable time. The explanations we used in the paper took a few minutes, which is too slow.
4. Thinking through what is needed to use LIME in regression problems. An obvious problem is that features with different scales make it really hard to interpret.
5. Figuring out better alternatives to discretizing the data for tabular data. Discretizing is definitely more interpretable, but we may just want to treat features as continuous.
6. Figuring out better ways to sample around a data point for tabular data. One example is sampling columns from the training set assuming independence, or some form of conditional sampling.
