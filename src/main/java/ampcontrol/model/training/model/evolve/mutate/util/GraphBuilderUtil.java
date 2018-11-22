package ampcontrol.model.training.model.evolve.mutate.util;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration.GraphBuilder;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.ElementWiseVertex;
import org.deeplearning4j.nn.conf.graph.GraphVertex;
import org.deeplearning4j.nn.conf.graph.LayerVertex;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer;

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.Optional;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.function.Predicate;

/**
 * Various hideous utility functions around {@link GraphBuilder} to assist in traversal.
 */
public class GraphBuilderUtil {

    /**
     * Returns a {@link LayerVertex} with the given name if one exists in the {@link GraphBuilder}.
     * Would very much prefer to do this some other way, but the API does not seem allow for it
     */
    public static final BiFunction<String, GraphBuilder, Optional<LayerVertex>> vertexAsLayerVertex =
            (layerName, graphBuilder) -> Optional.ofNullable(graphBuilder.getVertices().get(layerName))
                    .filter(gv -> gv instanceof LayerVertex)
                    .map(gv -> ((LayerVertex) gv));
    /**
     * Returns a {@link FeedForwardLayer} of a given {@link LayerVertex} if that is the type of layer it encapsulates
     * Would very much prefer to do this some other way, but the API does not seem allow for it
     */
    public static final BiFunction<String, Optional<LayerVertex>, Optional<FeedForwardLayer>> layerVertexAsFeedForward =
            (layerName, layerVertexOptional) -> layerVertexOptional
                    .map(LayerVertex::getLayerConf)
                    .map(NeuralNetConfiguration::getLayer)
                    .filter(layer -> layer instanceof FeedForwardLayer)
                    .map(layer -> (FeedForwardLayer) layer);

    /**
     * Returns true if a change of size (NOut or NIn) propagates to the (next or prev) layers (NIn or NOut).
     * For example: If NOut is changed for conv1 in case conv1 -> maxpool -> conv2, then it is conv2 which
     * needs to change its NIn as the maxpool is transparent w.r.t sizes. The converse goes for the case
     * when NIn is changed in conv2; then conv1 needs to change NOut for the very same reason.
     *
     * @param builder Configuration to check against
     * @return Predicate which is true if size change propagates
     */
    public static Predicate<String> changeSizePropagates(GraphBuilder builder) {
        return vertex -> Optional.ofNullable(builder.getVertices().get(vertex))
                .map(GraphBuilderUtil::doesSizeChangePropagate)
                .orElse(false);
    }

    /**
     * Returns true if the given vertex requires that all its inputs have the same size. So far, only the
     * {@link ElementWiseVertex} is known to have this property.
     *
     * @param builder Configuration to check against
     * @return Predicate which is true if size change propagates backwards
     */
    public static Predicate<String> changeSizePropagatesBackwards(GraphBuilder builder) {
        return vertex -> Optional.ofNullable(builder.getVertices().get(vertex))
                .map(GraphBuilderUtil::doesNOutChangePropagateToInputs)
                .orElse(false);
    }

    /**
     * Return the given vertex as a {@link FeedForwardLayer} if this is the type. Otherwise return empty.
     *
     * @param builder Configuration containing the sought vertex
     * @return Function mapping a vertex name to an Optional FeedForwardLayer.
     */
    public static Function<String, Optional<FeedForwardLayer>> asFeedforwardLayer(GraphBuilder builder) {
        return vertex -> GraphBuilderUtil.vertexAsLayerVertex
                .andThen(layerVertex -> GraphBuilderUtil
                        .layerVertexAsFeedForward.apply(vertex, layerVertex))
                .apply(vertex, builder);
    }

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

    /**
     * Gets the output size of a named layer without building the graph. Will recurse up the graph until
     * a {@link FeedForwardLayer} is found in case given layer is not of type {@link FeedForwardLayer}. Assumption made:
     * Only {@link FeedForwardLayer}s can have nIn != nOut.
     *
     * @param layerName    layer to determine output size for
     * @param graphBuilder Builder to search
     * @return the output size of the given layer
     */
    public static long getOutputSize(String layerName, GraphBuilder graphBuilder) {
        return findNextSize(layerName, graphBuilder, FeedForwardLayer::getNOut);
    }

    private static long findNextSize(
            String layerName,
            GraphBuilder graphBuilder,
            Function<FeedForwardLayer, Long> sizeMapping) {
        return asFeedforwardLayer(graphBuilder).apply(layerName).map(sizeMapping)
                .orElseGet(() ->
                        new ForwardOf(graphBuilder).children(layerName)
                                .mapToLong(inputLayerName -> findNextSize(inputLayerName, graphBuilder, FeedForwardLayer::getNIn))
                                .findAny()
                                .orElseThrow(() -> new IllegalStateException("Could not find any feedforward layers after " + layerName)));
    }

    /**
     * Gets the input size of a named layer without building the graph. Will recurse down the graph until
     * and sum up nOut of all found {@link FeedForwardLayer} in case given layer is not of type {@link FeedForwardLayer}.
     * Assumption made: Only {@link FeedForwardLayer}s can have nIn != nOut.
     *
     * @param layerName    layer to determine input size for
     * @param graphBuilder Builder to search
     * @return the input size of the given layer
     */
    public static long getInputSize(String layerName, GraphBuilder graphBuilder) {
        return sumPrevSize(layerName, graphBuilder, FeedForwardLayer::getNIn);
    }

    private static long sumPrevSize(String layerName,
                                    GraphBuilder graphBuilder,
                                    Function<FeedForwardLayer, Long> sizeMapping) {
        return asFeedforwardLayer(graphBuilder).apply(layerName).map(sizeMapping)
                .orElseGet(() -> Optional.of(new BackwardOf(graphBuilder).children(layerName)
                        .limit(graphBuilder.getVertices().get(layerName) instanceof MergeVertex ? Long.MAX_VALUE : 1)
                        .mapToLong(inputLayerName -> sumPrevSize(inputLayerName, graphBuilder, FeedForwardLayer::getNOut))
                        .sum())
                        .orElseGet(() -> graphBuilder.getNetworkInputTypes().stream()
                                .mapToLong(inputType -> inputType.getShape(false)[0])
                                .sum()));

    }

    /**
     * Creates a {@link TraverseBuilder} which is set up to provide children which can be used to compute size based on
     * inputs
     * @param builder Has the config to traverse
     * @return a {@link TraverseBuilder}
     */
    public static TraverseBuilder<String> inputSizeTravere(GraphBuilder builder) {
        final Deque<Long> limits = new ArrayDeque<>();
        return TraverseBuilder.backwards(builder)
                .enterCondition(vertex -> true)
                .enterListener(vertex -> {
                    if (builder.getVertices().get(vertex) instanceof ElementWiseVertex) {
                        limits.push(1L);
                    } else {
                        limits.push(Long.MAX_VALUE);
                    }
                })
                .leaveListener(vertex -> limits.pop())
                .limitTraverse(limits::peekFirst);
    }
}
