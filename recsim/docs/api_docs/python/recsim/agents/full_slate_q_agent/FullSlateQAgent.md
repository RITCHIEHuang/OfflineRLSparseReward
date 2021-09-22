<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="recsim.agents.full_slate_q_agent.FullSlateQAgent" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="begin_episode"/>
<meta itemprop="property" content="bundle_and_checkpoint"/>
<meta itemprop="property" content="end_episode"/>
<meta itemprop="property" content="step"/>
<meta itemprop="property" content="unbundle"/>
</div>

# recsim.agents.full_slate_q_agent.FullSlateQAgent

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

</table>

<a target="_blank" href="https://github.com/google-research/recsim/tree/master/recsim/agents/full_slate_q_agent.py">View
source</a>

A recommender agent implements full slate Q-learning based on DQN agent.

Inherits From:
[`DQNAgentRecSim`](../../../recsim/agents/dopamine/dqn_agent/DQNAgentRecSim.md),
[`AbstractEpisodicRecommenderAgent`](../../../recsim/agent/AbstractEpisodicRecommenderAgent.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>recsim.agents.full_slate_q_agent.FullSlateQAgent(
    sess, observation_space, action_space, optimizer_name='', eval_mode=False,
    **kwargs
)
</code></pre>

<!-- Placeholder for "Used in" -->

This is a standard, nondecomposed Q-learning method that treats each slate
atomically (i.e., holistically) as a single action.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`sess`
</td>
<td>
a Tensorflow session.
</td>
</tr><tr>
<td>
`observation_space`
</td>
<td>
A gym.spaces object that specifies the format of
observations.
</td>
</tr><tr>
<td>
`action_space`
</td>
<td>
A gym.spaces object that specifies the format of actions.
</td>
</tr><tr>
<td>
`optimizer_name`
</td>
<td>
The name of the optimizer.
</td>
</tr><tr>
<td>
`eval_mode`
</td>
<td>
A bool for whether the agent is in training or evaluation mode.
</td>
</tr><tr>
<td>
`**kwargs`
</td>
<td>
Keyword arguments to the DQNAgent.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`multi_user`
</td>
<td>
Returns boolean indicating whether this agent serves multiple users.
</td>
</tr>
</table>

## Methods

<h3 id="begin_episode"><code>begin_episode</code></h3>

<a target="_blank" href="https://github.com/google-research/recsim/tree/master/recsim/agents/full_slate_q_agent.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>begin_episode(
    observation
)
</code></pre>

Returns the agent's first action for this episode.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`observation`
</td>
<td>
numpy array, the environment's initial observation.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
An integer array of size _slate_size, the selected slated, each
element of which is an index in the list of doc_obs.
</td>
</tr>

</table>

<h3 id="bundle_and_checkpoint"><code>bundle_and_checkpoint</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>bundle_and_checkpoint(
    checkpoint_dir, iteration_number
)
</code></pre>

Returns a self-contained bundle of the agent's state.

This is used for checkpointing. It will return a dictionary containing all
non-TensorFlow objects (to be saved into a file by the caller), and it saves all
TensorFlow objects into a checkpoint file.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`checkpoint_dir`
</td>
<td>
str, directory where TensorFlow objects will be saved.
</td>
</tr><tr>
<td>
`iteration_number`
</td>
<td>
int, iteration number to use for naming the checkpoint
file.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A dict containing additional Python objects to be checkpointed by the
experiment. If the checkpoint directory does not exist, returns None.
</td>
</tr>

</table>

<h3 id="end_episode"><code>end_episode</code></h3>

<a target="_blank" href="https://github.com/google-research/recsim/tree/master/recsim/agents/full_slate_q_agent.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>end_episode(
    reward, observation
)
</code></pre>

Signals the end of the episode to the agent.

We store the observation of the current time step, which is the last observation
of the episode.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`reward`
</td>
<td>
float, the last reward from the environment.
</td>
</tr><tr>
<td>
`observation`
</td>
<td>
numpy array, the environment's initial observation.
</td>
</tr>
</table>

<h3 id="step"><code>step</code></h3>

<a target="_blank" href="https://github.com/google-research/recsim/tree/master/recsim/agents/full_slate_q_agent.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>step(
    reward, observation
)
</code></pre>

Receives observations of environment and returns a slate.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`reward`
</td>
<td>
A double representing the overall reward to the recommended slate.
</td>
</tr><tr>
<td>
`observation`
</td>
<td>
A dictionary that stores all the observations including:
- user: A list of floats representing the user's observed state
- doc: A list of observations of document features
- response: A vector valued response signal that represent user's
response to each document
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>

<tr>
<td>
`slate`
</td>
<td>
An integer array of size _slate_size, where each element is an
index in the list of document observvations.
</td>
</tr>
</table>

<h3 id="unbundle"><code>unbundle</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>unbundle(
    checkpoint_dir, iteration_number, bundle_dictionary
)
</code></pre>

Restores the agent from a checkpoint.

Restores the agent's Python objects to those specified in bundle_dictionary, and
restores the TensorFlow objects to those specified in the checkpoint_dir. If the
checkpoint_dir does not exist, will not reset the agent's state.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`checkpoint_dir`
</td>
<td>
str, path to the checkpoint saved by tf.Save.
</td>
</tr><tr>
<td>
`iteration_number`
</td>
<td>
int, checkpoint version, used when restoring the replay
buffer.
</td>
</tr><tr>
<td>
`bundle_dictionary`
</td>
<td>
dict, containing additional Python objects owned by
the agent.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
bool, True if unbundling was successful.
</td>
</tr>

</table>
