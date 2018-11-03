package ampcontrol.model.training.model.evolve.mutate;

import ampcontrol.model.training.model.evolve.mutate.util.*;
import lombok.Builder;
import lombok.Getter;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration.GraphBuilder;
import org.deeplearning4j.nn.conf.graph.GraphVertex;
import org.deeplearning4j.nn.conf.graph.LayerVertex;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.function.Function;
import java.util.function.UnaryOperator;
import java.util.stream.Collectors;
import java.util.stream.Stream;

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
        private final Map<String, OptionalLong> input = new HashMap<>();
        private final Map<String, OptionalLong> output = new HashMap<>();

        private boolean input(String name) {
            return input.containsKey(name);
        }

        private boolean output(String name) {
            return output.containsKey(name);
        }

        private OptionalLong prevNout(String name) {
            return input.get(name);
        }

        private void addInput(String name) {
            if (Optional.ofNullable(input.put(name, OptionalLong.empty())).isPresent()) {
                throw new IllegalStateException("Visited " + name + " twice!");
            }
        }

        private void addOutput(String name) {
            if (Optional.ofNullable(output.put(name, OptionalLong.empty())).isPresent()) {
                throw new IllegalStateException("Visited " + name + " twice!");
            }
        }

        private void setPrevNOut(String name, long nOut) {
            input.put(name, OptionalLong.of(nOut));
            // throw new IllegalStateException("Tried to reset previous nOut of " + name);
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

        final HasVistited visited = new HasVistited();
        visited.addInput(layerName);
        visited.setPrevNOut(layerName, oldNout);

        propagateNOutChange2(
                visited,
                builder,
                layerName,
                oldNout - layerConf.getNOut());
        return builder;
    }

    private void propagateNOutChange2(
            HasVistited visited,
            GraphBuilder builder,
            String layerName,
            final long deltaSize) {

        final Function<String, Optional<FeedForwardLayer>> asFf = GraphBuilderUtil.asFeedforwardLayer(builder);

        final Graph<String> forwardGraph = new TraverseForward(builder)
//                .enterListener(vertex -> System.out.println("\tHandle NOut change " + vertex + " with outputs: " + builder.getVertexInputs().entrySet().stream()
//                        .filter(entry -> entry.getValue().contains(vertex))
//                        .map(Map.Entry::getKey)
//                        .collect(Collectors.toSet())))
//                .leaveListener(vertex -> System.out.println("\tDone with NOut change " + vertex))
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


        final Map<String, Long> deltas = new HashMap<>();
        deltas.put(layerName, deltaSize);
        final Graph<String> backwardGraph = new TraverseBackward(builder)
                .enterListener(vertex -> {
                   // System.out.println("\tHandle NOut change backwards " + vertex + " with inputs: " + builder.getVertexInputs().get(vertex));
                    deltas.putAll(calcInputLayerDeltas(deltas, builder, vertex, deltas.getOrDefault(vertex, deltaSize)));
                })
                //.leaveListener(vertex -> System.out.println("\tDone with NOut change backwards " + vertex))
                // nOutDelta == 0 below might mask shortcoming of alg:
                // If you end up here with delta != original delta (e.g. change of size of original mutation) the deltas
                // are probably not correct as they might "compensate" for previous size changes in a way in which they
                // should not. So far, I think this can only happen when the "original" mutation is in a fork in which
                // case we know (?) that the output sizes are correct
                .visitCondition(inputName -> !visited.input(inputName) && deltas.get(inputName) != 0)
                .visitListener(inputName ->
                {
                    //System.out.println("\t\t Handle input layer " + inputName + " in context of " + layerName + " visited: " + visited.input(inputName) + " delta: " + nOutDelta);
                    asFf.apply(inputName)
                            .ifPresent(layer -> {
                                final long nOutDelta = deltas.get(inputName);
                                //System.out.println("\t\t Set nOut of layer " + inputName + " from " + layer.getNOut() + " to " + (layer.getNOut() - nOutDelta));
                                log.info("Set nOut of layer " + inputName + " from " + layer.getNOut() + " to " + (layer.getNOut() - nOutDelta));
                                //visited.setPrevNOut(inputName, layer.getNOut());
                                visited.addInput(inputName);
                                layer.setNOut(layer.getNOut() - nOutDelta);
                                if (changeNinMeansChangeNout(layer) && !visited.output(inputName)) {
                                    layer.setNIn(layer.getNIn() - nOutDelta);
                                    visited.addOutput(inputName);
                                }
                            });
                })
                .build();

        // Whatever comes out from backwardGraph which is a feedforward layer needs to go into
        // back into the "loop", starting again with the forwardGraph. Why? Don't remember,
        // old recursive code did this and testcases fail if it is not done. Will probably
        // have to look it up someday...
        new Traverse<>(
                vertex -> asFf.apply(vertex).isPresent(),
                new Connect<>(forwardGraph, backwardGraph)).children(layerName).count();
    }


    private Map<String, Long> calcInputLayerDeltas(Map<String, Long> deltas, GraphBuilder builder, String layerName, long deltaSize) {
        final List<String> inputs = builder.getVertexInputs().get(layerName);
        final GraphVertex vertex = builder.getVertices().get(layerName);
        final long[] layerSizes = new long[inputs.size()];

        if (vertex instanceof MergeVertex) {
            long remainder = deltaSize;
            Boolean[] validLayers = new Boolean[inputs.size()];
            for (int i = 0; i < validLayers.length; i++) {
                final String inputName = inputs.get(i);
                layerSizes[i] = GraphBuilderUtil.asFeedforwardLayer(builder).apply(inputName)
                        .map(FeedForwardLayer::getNOut)
                        .orElse(0L);
                validLayers[i] = layerSizes[i] > 0;
                if (validLayers[i]) {
                    remainder -= deltas.getOrDefault(inputName, 0L);
                }
            }

            for (int i = 0; i < validLayers.length; i++) {
                final String inputName = inputs.get(i);
                final long layerSizesSum = Arrays.stream(layerSizes, i, validLayers.length).sum();
                if (validLayers[i] && !deltas.containsKey(inputName)) {
                    final long delta = Math.min(layerSizes[i] - 1, Math.min((remainder * layerSizes[i]) / layerSizesSum, remainder));
                    deltas.put(inputs.get(i), delta);
                    remainder -= delta;
                } else if(!validLayers[i]){
                    deltas.put(inputs.get(i), 0L);
                }
            }

            if (Stream.of(validLayers).anyMatch(valid -> valid) && remainder != 0) {
                throw new IllegalStateException("Failed to distribute deltaSize over " + inputs + " deltas: " +
                        deltas + " layerSizes : " + Arrays.toString(layerSizes) + " remainder: " + remainder);
            }

        } else {
            deltas.putAll(inputs.stream().collect(Collectors.toMap(
                    name -> name,
                    name -> deltaSize
            )));
        }
        return deltas;
    }

    private long getMinNOut(GraphBuilder builder, String vertexName) {
        return new TraverseForward(builder).build().children(vertexName)
                .mapToLong(childName ->
                        new Filter<>(GraphBuilderUtil.changeSizePropagates(builder).negate(),
                                new TraverseBackward(builder)
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
