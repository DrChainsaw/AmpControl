package ampcontrol.model.training.model.evolve.mutate;

import ampcontrol.model.training.model.evolve.mutate.layer.LayerMutationInfo;
import lombok.Builder;
import lombok.Getter;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration.GraphBuilder;
import org.deeplearning4j.nn.conf.graph.ElementWiseVertex;
import org.deeplearning4j.nn.conf.graph.GraphVertex;
import org.deeplearning4j.nn.conf.graph.LayerVertex;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.function.UnaryOperator;
import java.util.stream.Collectors;

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

    public static class HasVistited {
        private final Map<String, OptionalLong> input = new HashMap<>();
        private final Map<String, OptionalLong> output = new HashMap<>();

        private boolean input(String name) {
            return input.containsKey(name);
        }

        private boolean output(String name) {
            return output.containsKey(name);
        }

        private OptionalLong prevNout(String name) {return input.get(name);}

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
            if(input.put(name, OptionalLong.of(nOut)).isPresent()) {
                throw new IllegalStateException("Tried to reset previous nOut of " + name);
            }
        }
    }

    public NoutMutation(Supplier<NoutMutationDescription> mutationLayerSupplier) {
        this.mutationLayerSupplier = mutationLayerSupplier;
    }

    @Override
    public GraphBuilder mutate(GraphBuilder builder) {
        mutationLayerSupplier.stream().forEach(mutation -> updateNoutOfLayer(new HasVistited(), builder, mutation));
        return builder;
    }

    private GraphBuilder updateNoutOfLayer(
            HasVistited visited,
            GraphBuilder builder,
            NoutMutationDescription mutation) {
        final String layerName = mutation.getLayerName();

        final FeedForwardLayer layerConf = (FeedForwardLayer) ((LayerVertex) builder.getVertices().get(layerName)).getLayerConf().getLayer();
        final long oldNout = layerConf.getNOut();
        final long newNout = Math.max(mutation.getMutateNout().apply(oldNout), getMinNOut(builder, layerName, new HasVistited()));
        layerConf.setNOut(newNout);
        visited.addInput(layerName);
        visited.setPrevNOut(layerName, oldNout);
        log.info("Mutating nOut of layer " + layerName + " from " + oldNout + " to " + layerConf.getNOut());

        //System.out.println("Mutating nOut of layer " + layerName + " from " + oldNout + " to " + layerConf.getNOut());
        propagateNOutChange(visited, builder, layerName, oldNout - layerConf.getNOut());
        //updateNinOfOutputLayer(builder, layerName);
         //System.out.println("Handled inputs: " + visited.input);
         //System.out.println("Handled outputs: " + visited.output);
        return builder;
    }

    private void propagateNOutChange(
            HasVistited visited,
            GraphBuilder builder,
            String layerName,
            long deltaSize) {

           //System.out.println("\tHandle NOut change " + layerName + " with outputs: " + builder.getVertexInputs().entrySet().stream()
//                   .filter(entry -> entry.getValue().contains(layerName))
//                   .map(Map.Entry::getKey)
//                   .collect(Collectors.toSet()));

        if (builder.getNetworkInputs().contains(layerName)) {
            return;
        }

        for (String outputName : builder.getVertexInputs().entrySet().stream()
                .filter(entry -> entry.getValue().contains(layerName))
                .map(Map.Entry::getKey)
                .collect(Collectors.toSet())) {

            if (visited.output(outputName)) {
                continue;
            }

            //System.out.println("\t\t Handle output layer: " + outputName + " in context of " + layerName);
            visited.addOutput(outputName);

            LayerMutationInfo.vertexAsLayerVertex
                    .andThen(layerVertex -> LayerMutationInfo
                            .layerVertexAsFeedForward.apply(outputName, layerVertex))
                    .apply(outputName, builder)
                    .ifPresent(layer -> {
                        //System.out.println("\t\t Set nIn of layer " + outputName + " from " + layer.getNIn() + " to " + (layer.getNIn() - deltaSize));
                        log.info("Set nIn of layer " + outputName + " from " + layer.getNIn() + " to " + (layer.getNIn() - deltaSize));
                        layer.setNIn(layer.getNIn() - deltaSize);
                        if (changeNinMeansChangeNout(layer) && !visited.input(outputName)) {
                            layer.setNOut(layer.getNOut() - deltaSize);
                            visited.addInput(outputName);
                        }
                    });
            final GraphVertex vertex = builder.getVertices().get(outputName);

            if (doesSizeChangePropagate(vertex)) {
                propagateNOutChange(
                        visited,
                        builder,
                        outputName,
                        deltaSize);
            }

            if (doesNOutChangePropagateToInputs(vertex)) {
                propagateNOutChangeToInputs(
                        visited,
                        builder,
                        outputName,
                        deltaSize);
            }
        }
         //System.out.println("\tDone with NOut change " + layerName);

    }

    private void propagateNOutChangeToInputs(
            HasVistited visited,
            GraphBuilder builder,
            String layerName,
            long deltaSize) {

        //System.out.println("\tHandle NOut change backwards " + layerName + " with inputs: " + builder.getVertexInputs().get(layerName));
        if (builder.getNetworkInputs().contains(layerName)) {
            return;
        }

        final long[] deltas = calcInputLayerDeltas(visited, builder, layerName, deltaSize);

        final List<String> inputs = builder.getVertexInputs().get(layerName);
        for (int i = 0; i < inputs.size(); i++) {
            final String inputName = inputs.get(i);
            final long nOutDelta = deltas[i];

            if (visited.input(inputName)) {
                continue;
            }

            //System.out.println("\t\t Handle input layer " + inputName + " in context of " + layerName);
            visited.addInput(inputName);

            LayerMutationInfo.vertexAsLayerVertex
                    .andThen(layerVertex -> LayerMutationInfo
                            .layerVertexAsFeedForward.apply(inputName, layerVertex))
                    .apply(inputName, builder)
                    .ifPresent(layer -> {
                        //System.out.println("\t\t Set nOut of layer " + inputName + " from " + layer.getNOut() + " to " + (layer.getNOut() - nOutDelta));
                        log.info("Set nOut of layer " + inputName + " from " + layer.getNOut() + " to " + (layer.getNOut() - nOutDelta));
                        visited.setPrevNOut(inputName, layer.getNOut());
                        layer.setNOut(layer.getNOut() - nOutDelta);
                        if (changeNinMeansChangeNout(layer) && !visited.output(inputName)) {
                            layer.setNIn(layer.getNIn() - nOutDelta);
                            visited.addOutput(inputName);
                        }
                        propagateNOutChange(
                                visited,
                                builder,
                                inputName,
                                deltaSize);
                    });

            if (doesSizeChangePropagate(builder.getVertices().get(inputName))) {
                propagateNOutChangeToInputs(
                        visited,
                        builder,
                        inputName,
                        nOutDelta);
            }
        }
        //System.out.println("\tDone with NOut change backwards " + layerName);
    }

    private long[] calcInputLayerDeltas(HasVistited visited, GraphBuilder builder, String layerName, long deltaSize) {
        final List<String> inputs = builder.getVertexInputs().get(layerName);
        final GraphVertex vertex = builder.getVertices().get(layerName);
        final long[] deltas = new long[inputs.size()];//getDeltaSizePerInputVertex(deltaSize, builder.getVertices().get(layerName), inputs.size());
        final long[] layerSizes = new long[inputs.size()];
        if(vertex instanceof MergeVertex) {
            long remainder = deltaSize;
            Boolean[] validLayers = new Boolean[inputs.size()];
            for(int i = 0; i < deltas.length; i++) {
                final String inputName = inputs.get(i);
                layerSizes[i] = LayerMutationInfo.vertexAsLayerVertex
                        .andThen(layerVertex -> LayerMutationInfo
                                .layerVertexAsFeedForward.apply(inputName, layerVertex))
                        .apply(inputName, builder)
                        .map(FeedForwardLayer::getNOut)
                        .orElse(0L);
                validLayers[i] = layerSizes[i] > 0;
                if(validLayers[i] && visited.input(inputName)) {
                    remainder += layerSizes[i] - visited.prevNout(inputName).orElse(layerSizes[i]);
                }
            }

            for(int i = 0; i < deltas.length; i++) {
                final String inputName = inputs.get(i);
                final long layerSizesSum = Arrays.stream(layerSizes, i, deltas.length).sum();
                if(validLayers[i] && !visited.input(inputName)) {
                    deltas[i] = Math.min(layerSizes[i] - 1, Math.min((remainder * layerSizes[i]) / layerSizesSum, remainder));
                    remainder -= deltas[i];
                }
            }

            if(remainder != 0) {
                throw new IllegalStateException("Failed to distribute deltaSize over " + inputs + " deltas: " +
                        Arrays.toString(deltas) + " layerSizes : " + Arrays.toString(layerSizes));
            }

        } else {
            Arrays.fill(deltas, deltaSize);
        }
        return deltas;
    }

    private long getMinNOut(GraphBuilder builder, String vertexName, HasVistited vistited) {

        vistited.addInput(vertexName);

        if (builder.getNetworkInputs().contains(vertexName)) {
            return 1;
        }

        long minNoutUp = 0;
        long minNoutDown = 0;
        for (String outputName : builder.getVertexInputs().entrySet().stream()
                .filter(entry -> entry.getValue().contains(vertexName))
                .map(Map.Entry::getKey)
                .collect(Collectors.toSet())) {

            if(vistited.output(outputName)) {
                continue;
            }

            vistited.addOutput(outputName);
            final GraphVertex outputVertex = builder.getVertices().get(outputName);
            if (doesSizeChangePropagate(outputVertex)) {
                minNoutUp = Math.max(minNoutUp, getMinNOut(builder, outputName, vistited));
            }

            if(doesNOutChangePropagateToInputs(outputVertex)) {
                minNoutDown += getMinNOutBackwards(builder, outputName, vistited);
            }
        }

        return Math.max(minNoutDown, minNoutUp);
    }

    private long getMinNOutBackwards(GraphBuilder builder, String vertexName, HasVistited vistited) {

        if (builder.getNetworkInputs().contains(vertexName)) {
            return 1;
        }

        long minNout = 0;
        for(String inputName: builder.getVertexInputs().get(vertexName)) {

            if(vistited.input(inputName)) {
                continue;
            }
            vistited.addInput(inputName);

            if (doesSizeChangePropagate(builder.getVertices().get(inputName))) {
                minNout += getMinNOutBackwards(builder, inputName, vistited);
            } else {
                minNout++;
            }
        }
        return minNout;
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

    /**
     * Returns true if a change of size (NOut or NIn) propagates to the (next or prev) layers (NIn or NOut).
     * For example: If NOut is changed for conv1 in case conv1 -> maxpool -> conv2, then it is conv2 which
     * needs to change its NIn as the maxpool is transparent w.r.t sizes.
     *
     * @param vertex Vertex to check for
     * @return true if size change propagates
     */
    private static boolean doesSizeChangePropagate(GraphVertex vertex) {
        if (!(vertex instanceof LayerVertex)) {
            return true;
        }
        LayerVertex layerVertex = (LayerVertex) vertex;

        if (layerVertex.numParams(false) == 0) {
            return true;
        }


        return layerVertex.getLayerConf().getLayer() instanceof BatchNormalization;

    }

    /**
     * Return true if a change in NOut propagates the NIn of all input layers.
     * Example of this is ElementWiseVertex: If one of the layers which is input
     * changes its NOut, the other input layer must also change its NOut
     *
     * @param vertex Vertex to check
     * @return True if change in NOut propagates to input layers
     */
    private static boolean doesNOutChangePropagateToInputs(GraphVertex vertex) {
        return vertex instanceof ElementWiseVertex;
    }
}
