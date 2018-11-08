package ampcontrol.model.training.model.evolve.mutate;

import ampcontrol.model.training.model.evolve.mutate.util.*;
import lombok.Builder;
import lombok.Getter;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration.GraphBuilder;
import org.deeplearning4j.nn.conf.graph.LayerVertex;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashSet;
import java.util.Optional;
import java.util.Set;
import java.util.function.Function;
import java.util.function.UnaryOperator;

/**
 * Mutation operation which changes nOut of layers
 *
 * @author Christian Skärby
 */
public class NoutMutation implements Mutation<ComputationGraphConfiguration.GraphBuilder> {

    private static final Logger log = LoggerFactory.getLogger(ampcontrol.model.training.model.evolve.mutate.NoutMutation.class);

    private final Supplier<NoutMutationDescription> mutationLayerSupplier;

    /**
     * Description for how to mutate nOut
     */
    @Builder
    @Getter
    public static class NoutMutationDescription {
        private final String layerName;
        private final UnaryOperator<Long> mutateNout;
    }

    private static class HasVistited {
        private final Set<String> input = new HashSet<>();
        private final Set<String> output = new HashSet<>();

        private boolean input(String name) {
            return input.contains(name);
        }

        private boolean output(String name) {
            return output.contains(name);
        }

        private void addInput(String name) {
            if (!input.add(name)) {
                throw new IllegalStateException("Visited " + name + " twice!");
            }
        }

        private void addOutput(String name) {
            if (!output.add(name)) {
                throw new IllegalStateException("Visited " + name + " twice!");
            }
        }
    }

    public NoutMutation(Supplier<NoutMutationDescription> mutationLayerSupplier) {
        this.mutationLayerSupplier = mutationLayerSupplier;
    }

    @Override
    public GraphBuilder mutate(GraphBuilder builder) {
        mutationLayerSupplier.stream().forEach(mutation -> updateNoutOfLayer(builder, mutation));
        return builder;
    }

    private GraphBuilder updateNoutOfLayer(
            GraphBuilder builder,
            NoutMutationDescription mutation) {
        final String layerName = mutation.getLayerName();

        final FeedForwardLayer layerConf = (FeedForwardLayer) ((LayerVertex) builder.getVertices().get(layerName)).getLayerConf().getLayer();
        final long oldNout = layerConf.getNOut();
        final long newNout = Math.max(mutation.getMutateNout().apply(oldNout), getMinNOut(builder, layerName));
        layerConf.setNOut(newNout);

        log.info("Mutating nOut of layer " + layerName + " from " + oldNout + " to " + layerConf.getNOut());

        propagateNOutChange(
                builder,
                layerName,
                oldNout - layerConf.getNOut());
        return builder;
    }

    /**
     * Propagate a change in nOut to the next layer(s), i.e change nIn with the given delta size so that nOut of the
     * mutated layer is equal to nIn of all layers which have it as input. Why not just set all nIns to zero and let
     * the {@link GraphBuilder} handle it automatically? Because of nr 1 below basically.
     * <br><br>
     * <b>Brainf*ck level 0:</b> Next vertex might be of a type which does not allow nIn to be set or has constraint
     * nIn == nOut. We must propagate the change through to the next vertex which allows setting nIn and allows
     * nIn != nOut, hereafter referred to as a "termination layer" (as in "terminates the traversal").
     * <br><br>
     * <b>Brainf*ck level 1:</b> On the way to a termination layer a vertex is encountered which has multiple inputs and
     * which requires that all its inputs are of the same size (i.e have the same nOut). We must traverse backwards from
     * this vertex to find all termination layers which leads to its inputs. Example of such a vertex is
     * ElementWiseVertex which is commonly used to create residual blocks.
     * <br><br>
     * <b>Brainf*ck level 2:</b> While traversing backwards the outputs of each encountered vertex may have multiple
     * outputs. We must propagate the change forward through any not yet encountered output vertexes in the exact same
     * manner as done for the mutated layer to begin with.
     * <br><br>
     * <b>Brainf*ck level 3:</b> If a {@link MergeVertex} is encountered while traversing backwards the delta size
     * applies to the sum of its input vertices, and not to each one of them individually. Distribute the delta size
     * across them and continue the traversal down each branch using the recomputed delta size.
     * <br><br>
     * <b>Brainf*ck level 4:</b> While handling brainf*ck level 3, we might have changed nOut of one or more vertices
     * already without having traversed the other paths. This must be taken into account when distributing the delta
     * sizes. I suspect that this can only happen in case the original target mutation layer is part of a fork which
     * touches an ElementWiseVertex.
     * <br><br>
     * <b>Brainf*ck level 5:</b> TBD (To Be Discovered)...
     *
     * @param builder   {@link GraphBuilder} to mutate
     * @param layerName Name of layer for which nOut has been changed
     * @param deltaSize How much the size of any dependent layers shall be changed.
     */
    private static void propagateNOutChange(
            GraphBuilder builder,
            String layerName,
            final long deltaSize) {

        final HasVistited visited = new HasVistited();
        visited.addInput(layerName);

        final Graph<String> forwardGraph = getForwardGraph(builder, deltaSize, visited);
        final Graph<String> backwardGraph = getBackwardGraph(builder, layerName, deltaSize, visited);

        // Whatever comes out from backwardGraph which is a feedforward layer needs to go into
        // back into the "loop", starting again with the forwardGraph due to brainf*ck level 2
        // described above. Why only feedforward layers? Answer: Those are the only ones for
        // which nOut was actually changed.
        final Function<String, Optional<FeedForwardLayer>> asFf = GraphBuilderUtil.asFeedforwardLayer(builder);
        new Traverse<>(
                vertex -> asFf.apply(vertex).isPresent(),
                new Connect<>(forwardGraph, backwardGraph)).children(layerName).forEachOrdered(residual -> {/* ignore */});
    }

    private static Graph<String> getForwardGraph(GraphBuilder builder, long deltaSize, HasVistited visited) {
        final Function<String, Optional<FeedForwardLayer>> asFf = GraphBuilderUtil.asFeedforwardLayer(builder);
        return TraverseBuilder.forwards(builder)
                //                .enterListener(vertex -> System.out.println("\tHandle NOut change " + vertex + " with outputs: " + builder.getVertexInputs().entrySet().stream()
                //                        .filter(entry -> entry.getValue().contains(vertex))
                //                        .map(Map.Entry::getKey)
                //                        .collect(Collectors.toSet())))
                //                .leaveListener(vertex -> System.out.println("\tDone with NOut change " + vertex))
                .visitCondition(outputName -> !visited.output(outputName))
                .visitListener(outputName -> asFf.apply(outputName)
                        .ifPresent(layer -> {
                            //                           System.out.println("\t\t Set nIn of layer " + outputName + " from " + layer.getNIn() + " to " + (layer.getNIn() - deltaSize));
                            log.info("Set nIn of layer " + outputName + " from " + layer.getNIn() + " to " + (layer.getNIn() - deltaSize));
                            layer.setNIn(layer.getNIn() - deltaSize);
                            if (changeNinMeansChangeNout(layer) && !visited.input(outputName)) {
                                layer.setNOut(layer.getNOut() - deltaSize);
                                visited.addInput(outputName);
                            }
                        }))
                .build();
    }

    private static Graph<String> getBackwardGraph(GraphBuilder builder, String layerName, long deltaSize, HasVistited visited) {

        final SizeVisitor deltaRegistry = new SizeVisitor(
                builder,
                deltaSize,
                (layerSize, delta) -> Math.min(layerSize-1, delta));
        deltaRegistry.set(layerName, deltaSize);

        final Function<String, Optional<FeedForwardLayer>> asFf = GraphBuilderUtil.asFeedforwardLayer(builder);
        // Just a note: Listeners below are a bit less insane than it might appear. The enterListener is not invoked
        // through in a Stream#peek call, but rather just before a stream is created (and immediately consumed) as the
        // Traverse graph creates a new collection for each recursion. This also makes the visitListener bit less of an
        // API-abuse as it does not touch any state which may impact it before the stream is consumed. One might (and
        // perhaps should) argue that the fact I had to write this comment to myself is reason enough to change the
        // design...

        return TraverseBuilder.backwards(builder)
                .enterListener(deltaRegistry::visit)
                //.leaveListener(vertex -> System.out.println("\tDone with NOut change backwards " + vertex))
                // nOutDelta == 0 below might mask shortcoming of alg:
                // If you end up here with delta != original delta (e.g. change of size of original mutation) the deltas
                // are probably not correct as they might "compensate" for previous size changes in a way in which they
                // should not. So far, I think this can only happen when the "original" mutation is in a fork in which
                // case we know (?) that the output sizes are correct
                // Possible candidate for Brainf*ck level 5...
                .visitCondition(inputName -> !visited.input(inputName) && deltaRegistry.getSize(inputName) != 0)
                .visitListener(inputName ->
                {
                    //System.out.println("\t\t Handle input layer " + inputName + " in context of " + layerName + " visited: " + visited.input(inputName) + " delta: " + nOutDelta);
                    asFf.apply(inputName)
                            .ifPresent(layer -> {
                                final long nOutDelta = deltaRegistry.getSize(inputName);
                                //System.out.println("\t\t Set nOut of layer " + inputName + " from " + layer.getNOut() + " to " + (layer.getNOut() - nOutDelta));
                                log.info("Set nOut of layer " + inputName + " from " + layer.getNOut() + " to " + (layer.getNOut() - nOutDelta));
                                visited.addInput(inputName);
                                layer.setNOut(layer.getNOut() - nOutDelta);
                                if (changeNinMeansChangeNout(layer) && !visited.output(inputName)) {
                                    layer.setNIn(layer.getNIn() - nOutDelta);
                                    visited.addOutput(inputName);
                                }
                            });
                })
                .build();
    }

    private long getMinNOut(GraphBuilder builder, String vertexName) {
        return TraverseBuilder.forwards(builder).build().children(vertexName)
                .mapToLong(childName ->
                        new Filter<>(GraphBuilderUtil.changeSizePropagates(builder).negate(),
                                TraverseBuilder.backwards(builder)
                                        .visitCondition(vertex -> !vertex.equals(vertexName))
                                        .build())
                                .children(childName).count())
                .max()
                .orElse(0);
    }

    /**
     * Returns true if the given layer is of a type where NIn and NOut must both be set to the same value
     *
     * @param layer Layer to check
     * @return true if the given layer is of a type where NIn and NOut must both be set to the same value
     */
    private static boolean changeNinMeansChangeNout(FeedForwardLayer layer) {

        // Is there any parameter which can tell this instead of hardcoding it to types like this?
        return layer instanceof BatchNormalization;
    }
}
