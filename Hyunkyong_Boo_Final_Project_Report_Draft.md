ARC-AGI Final Project Report
Hyunkyong Boo
hkbbooo@gatech.edu

Abstract- This report describes the final state of my ARC agent after the Milestone B, C, and D development cycle. The final agent is a deterministic, rule-based system that tests candidate transformations against all training pairs and applies a rule to the test grid only if it reproduces every visible output exactly. Over time, the agent expanded from generic geometric operations into a broader structural pipeline that includes panel composition, separator detection, connected-component analysis, enclosed-region reasoning, motif completion, and lightweight learned templates. On the final evaluation, the agent solved all 48 training problems and 36 of 48 hidden test problems, for a total of 84 out of 96. This report explains the design of the agent, its final performance, representative successes and failures, how the design evolved, and how its reasoning differs from human problem solving.

1 AGENT DESCRIPTION

My final agent is best described as a cascading rule engine with exact training validation. For each ARC problem, the agent loads every training input-output pair as a NumPy array and then evaluates a sequence of candidate transformations. A transformation is considered valid only if it reproduces every training output exactly. If a rule passes that check, the agent applies it to the test input and stores the result as a candidate prediction. The agent returns up to three unique predictions, which allows it to preserve more than one plausible interpretation on ambiguous tasks without resorting to random choice.

This design kept the system interpretable throughout the project. Every prediction comes from an explicit transformation that I can inspect, debug, and explain. The cost of that interpretability is that coverage depends on the breadth and quality of the rule library. If the correct abstraction is not represented in the library, the agent has no mechanism to invent it on the fly. As a result, the project became an exercise in expanding the library from simple image-wide transforms toward more object-centered and structure-sensitive reasoning.

1.1 Overall architecture

The top of the pipeline contains a list of direct rules. Some of these are generic transformations such as identity, rotation, reflection, transpose, cropping, and simple panel composition. Others are more specialized rules that recognize structural motifs: overlaying multiple equal panels, drawing diagonals through a seed, completing a missing subpanel using separator-based mirroring, filling enclosed regions, reflecting content inside a border, converting a marker pattern into a compact summary grid, and fitting detached pieces into a base layout.

After this direct rule layer, the agent tries a small number of lightweight learned templates. These are still deterministic rather than statistical. For example, the agent can infer a single-color extraction rule, a marker-shape rule based on a compact signature, a boolean combination rule for panel layouts, or a role-based hole-filling rule when the training examples indicate that certain colors play consistent structural roles. These routines do not optimize parameters or search over a large space of programs. Instead, they infer a narrow transformation schema from the training pairs and then apply that schema once to the test input.

Finally, for a few especially ambiguous tasks, the agent can emit extra hidden-test hypotheses. These are alternate deterministic interpretations of a task family that still preserve the visible training fit. This feature was added because several problems were locally solved but still failed hidden tests, which suggested that the visible examples underdetermined the true rule. Rather than replacing one fitted rule with another, the agent can now preserve several structurally related candidates and return up to three predictions.

1.2 Rule families

By the end of the project, the rule library clustered into several broad families.

The first family is geometric propagation. These rules extend a motif once a seed or anchor is identified. Examples include drawing both diagonals through a single colored seed, extending a line between two endpoints, moving a marker one step toward another marker, or reflecting content around a frame or border.

The second family is panel and separator reasoning. These rules detect full separator rows or columns, divide the grid into repeated subpanels, and then combine, mirror, overlay, or complete those panels. This family became especially important in Milestones C and D because many tasks are easier to solve once the grid is segmented into panels rather than treated as one undifferentiated image.

The third family is object and region reasoning. These rules rely on connected components, bounding boxes, flood fill, or enclosed-region detection. They allow the agent to distinguish inside from outside, identify closed barriers, recognize hole-bearing shapes, and reason about how objects should be mirrored or summarized.

The fourth family is compression and summarization. These rules map a larger structure to a smaller output by counting, cropping, recoloring, or building a compact summary grid. Problems in this family often require the agent to stop treating the task as “draw more pixels” and instead treat it as “extract a symbolic description of the input.”

1.3 Prediction flow

At a procedural level, the agent works in the following order. It first reads all training examples for a problem. It then tests each candidate rule using an exact equality check on every training input-output pair. Rules that fail even one example are rejected immediately. Rules that pass are applied to the test input, and duplicate predictions are removed. The agent next tries a few learned templates and panel-logic variants. Finally, if the problem belongs to a small set of ambiguous cases for which multiple hypotheses were intentionally preserved, the agent appends those extra alternatives and returns the first three unique predictions.

This means the agent’s behavior is conservative. It does not guess blindly. If a rule cannot justify itself on the visible training examples, it never reaches the test input. The advantage is that every prediction is grounded in the training data. The disadvantage is that the agent can still overfit by discovering a rule that matches the training pairs for the wrong reason. Many of the remaining hidden-test failures are best understood as exactly that kind of overfitting.

2 AGENT PERFORMANCE

The final agent solved all 48 visible training problems and 36 of 48 hidden test problems. Taken together, that is 84 solved problems out of 96 total, which corresponds to an overall performance score of 87.5%.

The main summary statistics are straightforward. The agent solved 48 of 48 training problems, or 100 percent, and 36 of 48 hidden test problems, or 75 percent. Combined, that is 84 of 96 total problems, or 87.5 percent overall.

The set-by-set breakdown highlights where the remaining weakness lies. On Set B, the agent solved 29 of 32 overall; on Set C, 30 of 32; and on Set D, 25 of 32. Because all visible problems were solved, the remaining gap is not local coverage but generalization. Set D produced the most hidden failures, which is consistent with its greater reliance on multi-step structural reasoning, role assignment, and ambiguity resolution.

The evaluation output also reported a total runtime of 0.400624 seconds across all 96 problems, or about 0.00417 seconds per problem. Efficiency was therefore not a bottleneck. The main limitation was representational: the agent can test many rules quickly, but it still fails when the correct abstraction is absent or too narrow.

3 AGENT SUCCESSES

The final agent performs best on tasks with a single dominant structural relation that can be expressed explicitly and validated strictly. In successful cases, the training examples are sufficient to isolate one strong rule, and that rule generalizes cleanly to the hidden test.

3.1 Problem 623ea044

Problem 623ea044 is a good example of a task where the agent’s geometric propagation strategy works well. In this problem, the input contains a single nonzero seed cell embedded in an otherwise empty grid. The correct output is produced by drawing both diagonals through that seed in the same color, continuing until the diagonals reach the boundary of the grid.

The agent solves this task by first verifying that there is exactly one nonzero cell in the input. Once it identifies that cell, it treats it as the anchor point for a diagonal propagation rule. It constructs an empty output grid of the same size and then marks every position on the two diagonals passing through the seed. Because the transformation depends only on the seed’s coordinates and the grid boundaries, it generalizes very cleanly across training and test examples of different sizes.

This problem illustrates one of the agent’s strongest design patterns: once a problem can be reframed as “identify a small anchor and propagate a rigid geometric relation from it,” the solution is both fast and reliable. There is little ambiguity about object roles, and there are few opportunities for spurious rule matches.

3.2 Problem cf98881b

Problem cf98881b shows a different strength: panel decomposition and ordered overlay. The input can be divided into three equal subpanels separated by full separator columns. The output is not a transformed version of the whole grid at once; instead, it is the result of overlaying the three panels with left-to-right priority on nonzero cells.

The agent’s reasoning process for this task is explicitly structural. It first detects full columns of a single separator color and uses them to split the grid into equal-width panels. It then checks that the three panels have compatible dimensions. After segmentation, the agent overlays the panels in order. The left panel provides the initial state, the middle panel fills any remaining zero cells, and the right panel fills whatever zeros remain after that. This implements a left-to-right nonzero-priority rule.

What makes this a useful success case is that the agent is not merely matching a visual template. It is identifying separators, using those separators to infer a panelized representation, and then applying an ordered combination rule at the panel level. That kind of reasoning is closer to symbolic decomposition than to global image transformation, and it is one of the places where the project clearly moved beyond simple rotation-and-flip baselines.

3.3 Problem c8b7cc0f

Problem c8b7cc0f demonstrates the agent’s summarization ability. In this task, the input contains a frame of one color and a small set of marker cells of another color. The output is a compact 3x3 summary grid rather than a transformed image of the original layout.

The agent solves this problem by first identifying the frame cells and computing the bounding box of that frame. It then counts how many marker-color cells appear inside the frame. Instead of preserving the original geometry, it converts that count into a row-major fill of a 3x3 output grid using the marker color. In effect, the input is treated as a structured count that should be compressed into a symbolic summary.

This is an important success because it highlights that the final agent is not limited to “draw more of the same pattern.” It can also map a larger arrangement to a smaller output when the training pairs suggest a counting or encoding rule. In human terms, this is the type of problem where a person might say, “the output is a summary of how many marked cells were present.” The agent does not phrase it that way, but the transformation it applies is functionally similar.

3.4 Broader success pattern

Across solved problems, the same strengths recur. The agent performs best when there is a stable object vocabulary, such as separators, borders, connected shapes, repeated panels, seed points, or compact summaries, and when the visible examples constrain the transformation tightly enough to rule out alternate explanations.

4 AGENT STRUGGLES

The final failures were concentrated in tasks where the visible training pairs permit more than one plausible rule, especially when the correct rule depends on object roles rather than literal color values or on a multi-stage structural decision that is easy to approximate but hard to infer exactly.

4.1 Problem b1948b0a

Problem b1948b0a is a clear example of hidden-test overfitting. On the visible examples, the most obvious interpretation is that one color should be rewritten as color 2 while another color should remain unchanged. In the public training examples, the transformation is perfectly consistent with the literal rule “replace 6 with 2 and keep 7.”

The problem is that the visible examples do not prove that the value 6 itself is intrinsically special. They only prove that, in the visible data, the cells currently colored 6 are the cells that must become 2 in the output. A human would immediately recognize that those are different claims. My agent, however, only has access to the explicit rules I give it. Because the visible examples were fully explained by a literal color remap, the agent could easily fit the training pairs while missing the deeper relational rule, if one exists.

The hidden failure strongly suggests that the real task is role-based rather than color-value-based. In other words, the correct transformation may be “change the cells playing role X into role color 2,” not “change all 6s into 2.” The agent has some limited mechanisms for role inference, but in this case they were not strong enough to disambiguate the hidden version. This failure captures one of the central weaknesses of the final design: exact fit is not the same as correct abstraction.

4.2 Problem d931c21c

Problem d931c21c exposes a different weakness: inferring inside, outside, and hole-filling roles precisely. In the public examples, the task involves hole-bearing shapes, an outer added ring, an inner fill color, and regions that must remain background. My final agent does have dedicated logic for this family. It detects connected components, identifies enclosed holes using flood fill, adds an outer ring around qualifying components, and paints some inner cells a different color.

However, the hidden failure indicates that the current implementation still approximates the true rule too narrowly. The difficult part is not simply discovering that there is a ring and a hole fill. The difficult part is deciding exactly which cells inside the component should become the inner color and which cells should remain background. The public examples support several similar interpretations, such as painting all hole cells, painting only the first interior layer, or painting cells adjacent to the original shape under a particular neighborhood definition.

My agent currently handles this by using a fixed adjacency-based criterion after identifying holes. That works on the public examples and therefore passes visible validation, but the hidden failure suggests that the true reasoning process depends on a more exact understanding of inside-versus-outside structure than my current implementation captures. This is a representative failure for the entire D subset: the agent can often recognize the right family of task but still miss the exact relational boundary that the hidden test is using.

4.3 Problem 992798f6

Problem 992798f6 illustrates a third weakness: path construction under ambiguous local orientation. The input contains one cell of color 1 and one cell of color 2 in an otherwise empty grid. The output adds a path of color 3 between them, but that path is not simply the shortest straight line in every case. Instead, the shape of the path depends on the orientation of the endpoints and the way horizontal, vertical, and diagonal segments are arranged.

My agent contains several variants of a path-tracing rule for this family. It can vary which endpoint is treated as the start, whether the first move is forced diagonal, and whether the path prioritizes one stepping pattern or another. Those variants were enough to fit the public examples, but the hidden failure indicates that the agent still is not truly inferring the path rule from the endpoints’ local roles.

The deeper issue is that the current agent chooses among a few hand-specified path procedures rather than reading a richer structural cue from the grid. A human looking at the examples would try to infer which endpoint “owns” the long horizontal or vertical segment and which endpoint anchors the diagonal portion. My agent does not infer that role directly; it only tries several candidate procedures and keeps the ones that fit the visible examples. Once again, that is enough for the public tasks but not robust enough for all hidden variants.

4.4 Broader failure pattern

The remaining hidden failures follow a common pattern. In most cases the agent captured part of the right idea, but the fitted rule remained too attached to a specific visible geometry, a fixed color role, or one narrow ordering of operations. That is the final ceiling of the project: broad coverage with incomplete robustness under hidden reparameterization.

5 AGENT DEVELOPMENT

The overall development pattern of my agent is best described as an expanding suite of explicit heuristics that became progressively more structural over time.

At the beginning of the project, the agent relied mostly on generic grid-wide transforms such as identity, rotations, reflections, transpose, cropping, and a few simple line-drawing or recoloring rules. This was enough for easier tasks, but it quickly became clear that many ARC problems are not solved by a single global transform.

In the middle phase of development, I shifted toward component-based and panel-based reasoning. I added helpers for connected components, neighborhood traversal, flood fill, panel segmentation, and nonzero overlay. This made it possible to distinguish objects from background, reason about subgrids separately, and detect enclosed regions.

During the later Milestone C and D work, the development process became more iterative and task-family oriented. Instead of treating each failure as an isolated case, I tried to identify recurring families such as separator completion, border reflection, piece fitting, hole filling, panel overlay, and motif summarization. That shift improved the agent because one abstraction could solve several tasks, but it also exposed a new challenge: some rules generalized across visible examples while still missing the exact invariant required by the hidden test.

This project also changed the way I think about “progress” in a rule-based ARC system. Early on, progress mostly meant adding more rules. By the end, progress meant improving representations: better object extraction, clearer role assignment, and more principled disambiguation among multiple fitted rules. One especially important late change was separating visible-example fit from hidden-test robustness. Once I reached full visible coverage, the remaining challenge was not discovering more transformations, but recognizing when a rule matched the training pairs for the wrong reason.

The final agent is still heuristic-heavy, but it is not just a bag of isolated tricks. It is a layered deterministic system whose major components are grid transforms, structural segmentation, connected-component reasoning, learned templates, and a small hidden-hypothesis mechanism for ambiguous tasks. In that sense, the development trajectory of the project was from surface manipulation toward more explicit relational reasoning, even though the implementation remained fully symbolic and hand-designed.

If I were continuing beyond this final submission, the next development step would not be simply adding dozens more task-specific rules. It would be reorganizing the agent around a stronger intermediate representation of objects, roles, and relations. Many of the remaining failures suggest that the current pipeline sees the right pieces but still lacks the abstraction layer needed to express “this color is functioning as the boundary,” “this interior is second-order rather than first-order,” or “these two endpoints play different structural roles.”

6 AGENT-HUMAN COMPARISON

At a high level, my agent and a human approach ARC in similar ways. Both look for repeated motifs, separators, symmetry, enclosed regions, and object boundaries. When my agent succeeds, it often succeeds for the same broad reason a human would: it isolates a structural cue and then applies a consistent relation. In Problem cf98881b, for example, both treat the grid as three panels whose output is an ordered composition. In Problem 623ea044, both identify a single seed and propagate a geometric relation from that anchor.

The difference is that the human can infer abstractions much more flexibly. A person is comfortable treating color as a role rather than a literal value, recognizing that a distorted variant may still instantiate the same concept, or revising an interpretation after only one contradictory example. My agent cannot do that unless I have already encoded the relevant abstraction as a rule or template. It does not invent a new representation during inference. It only checks whether one of its existing representations fits the evidence. This gap is especially visible in failures such as b1948b0a, where a person would naturally ask whether the important fact is “this is color 6” or “this is the color playing a structural role.”

Another major difference is how ambiguity is handled. A human can entertain several hypotheses and rank them using informal notions such as simplicity, semantic coherence, or analogy with previous tasks. My agent handles ambiguity procedurally. It preserves several candidate predictions only when I have explicitly chosen to include alternate hypotheses for a task family, and even then the alternatives come from predefined deterministic rules. Problem 992798f6 is a good example: a human can infer which endpoint should control the diagonal part of the path, while my agent cycles through a few hand-specified path procedures and keeps whichever ones fit the visible examples.

The agent also differs from a human in how brittle its attention is. A person can decide that a particular marker is irrelevant noise or that two different colors play the same structural role in different examples. My agent tends to treat surface details literally unless a rule has been written to abstract them away. This is why hidden-test failures often arise even after perfect visible performance. In d931c21c, for example, a human can reason in terms of nested regions and interior layers even if the exact colors shift, whereas my agent can recognize the task family but still misidentify which cells belong to the correct relational layer.

That said, the final system is not completely unlike human reasoning. It is not performing blind brute-force search across huge program spaces. It is closer to an explicit, restricted version of asking whether a problem is about separators, symmetry, or inside-versus-outside structure. The difference is that a human can create new categories on demand, while the agent can only choose among categories I already built.

7 CONCLUSION

The final agent achieved strong visible coverage and respectable hidden performance: 48 out of 48 training problems, 36 out of 48 hidden test problems, and 84 out of 96 overall. The project successfully pushed the agent from simple geometric manipulation toward more explicit structural reasoning over panels, borders, holes, repeated motifs, and compact summaries. The strongest parts of the final system are its interpretability, speed, and ability to validate a rule strictly across all training examples before making a prediction.

The main limitation is that a deterministic rule library can still overfit the visible examples even when every prediction is explainable. The final hidden failures are therefore informative: they show where the current abstractions remain too narrow, especially in role assignment, path construction, and multi-stage region reasoning. If I were to continue the project, the next major step would be to preserve the interpretability of the current system while introducing a richer object- and relation-centered representation that reduces reliance on literal colors and one-off geometric layouts.
