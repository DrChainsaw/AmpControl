package ampcontrol.model.training.model.evolve.mutate;

import ampcontrol.model.training.model.evolve.mutate.layer.LayerMutationInfo;
import lombok.Builder;
import lombok.Getter;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration.GraphBuilder;
import org.deeplearning4j.nn.conf.graph.ElementWiseVertex;
import org.deeplearning4j.nn.conf.graph.GraphVertex;
import org.deeplearning4j.nn.conf.graph.LayerVertex;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashSet;
import java.util.Map;
import java.util.Set;
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
        layerConf.setNOut(mutation.getMutateNout().apply(oldNout));
        visited.addInput(layerName);
        log.info("Mutating nOut of layer " + layerName + " from " + oldNout + " to " + layerConf.getNOut());

        propagateNOutChange(visited, builder, layerName, oldNout - layerConf.getNOut());
        //updateNinOfOutputLayer(builder, layerName);
        // System.out.println("Handled inputs: " + visited.input);
        // System.out.println("Handled outputs: " + visited.output);
        return builder;
    }

    private void propagateNOutChange(
            HasVistited visited,
            GraphBuilder builder,
            String layerName,
            long deltaSize) {

        //   System.out.println("Handle NOut change " + layerName);
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

            //   System.out.println("\t Handle output layer: " + outputName + " in context of " + layerName);
            visited.addOutput(outputName);

            LayerMutationInfo.vertexAsLayerVertex
                    .andThen(layerVertex -> LayerMutationInfo
                            .layerVertexAsFeedForward.apply(outputName, layerVertex))
                    .apply(outputName, builder)
                    .ifPresent(layer -> {
                        //  System.out.println("\t Set nIn of layer " + outputName + " from " + layer.getNIn() + " to " + (layer.getNIn() - deltaSize));
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
        // System.out.println("Done with NOut change " + layerName);

    }

    private void propagateNOutChangeToInputs(
            HasVistited visited,
            GraphBuilder builder,
            String layerName,
            long deltaSize) {

        //System.out.println("Handle NOut change backwards " + layerName);
        if (builder.getNetworkInputs().contains(layerName)) {
            return;
        }

        for (String inputName : builder.getVertexInputs().get(layerName)) {

            if (visited.input(inputName)) {
                continue;
            }

            //System.out.println("\t Handle input layer " + inputName + " in context of " + layerName);
            visited.addInput(inputName);

            LayerMutationInfo.vertexAsLayerVertex
                    .andThen(layerVertex -> LayerMutationInfo
                            .layerVertexAsFeedForward.apply(inputName, layerVertex))
                    .apply(inputName, builder)
                    .ifPresent(layer -> {
                        // System.out.println("\t Set nOut of layer " + inputName + " from " + layer.getNOut() + " to " + (layer.getNOut() - deltaSize));
                        layer.setNOut(layer.getNOut() - deltaSize);
                        if (changeNinMeansChangeNout(layer) && !visited.output(inputName)) {
                            layer.setNIn(layer.getNIn() - deltaSize);
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
                        deltaSize);
            }
        }
        //System.out.println("Done with NOut change backwards " + layerName);
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
