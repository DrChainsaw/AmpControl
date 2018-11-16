package ampcontrol.model.training.model.evolve.transfer;

import ampcontrol.model.training.model.evolve.mutate.util.Graph;
import ampcontrol.model.training.model.evolve.mutate.util.TraverseBuilder;
import lombok.AllArgsConstructor;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.GraphVertex;
import org.deeplearning4j.nn.graph.vertex.impl.MergeVertex;
import org.deeplearning4j.nn.params.BatchNormalizationParamInitializer;
import org.deeplearning4j.nn.params.CenterLossParamInitializer;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.*;
import java.util.function.Function;
import java.util.function.IntUnaryOperator;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Transfers parameters from one {@link ComputationGraph} to another. Shapes must not be identical.
 *
 * @author Christian Skärby
 */
public class ParameterTransfer {

    private final static String W = DefaultParamInitializer.WEIGHT_KEY;

    // private static final Logger log = LoggerFactory.getLogger(ParameterTransfer.class);

    private final ComputationGraph sourceCompGraph;
    private final Function<String, Optional<Function<Integer, Comparator<Integer>>>> compFactory;
    private final Set<String> nInTransferredFromPreviousNoutLayers = new HashSet<>();
    private final TransferRegistry registry;
    private final Map<String, MergeContext> mergeTasks = new LinkedHashMap<>();

    @AllArgsConstructor
    private static final class TransferContext {
        private final long[] sourceShape;
        private final long[] targetShape;
        private final int inputDimension;
        private final int outputDimension;
    }

    private static final class MergeContext {
        private final List<TransferContext> contexts = new ArrayList<>();
        private final MergeTransferBuffer.Builder builder;

        private MergeContext() {
            this.builder = MergeTransferBuffer.builder();
        }

        private void add(TransferContext context, TransferTask.ListBuilder inputBuilder) {
            // Validate input
            contexts.forEach(otherContext -> {
                if (otherContext.sourceShape.length != context.sourceShape.length) {
                    throw new IllegalArgumentException("Incorrect source shape! other: " +
                            Arrays.toString(otherContext.sourceShape) + " new  " +
                            Arrays.toString(context.sourceShape));
                }

                if (otherContext.targetShape.length != context.targetShape.length) {
                    throw new IllegalArgumentException("Incorrect target shape! other: " +
                            Arrays.toString(otherContext.targetShape) + " new  " +
                            Arrays.toString(context.targetShape));
                }


                if (otherContext.inputDimension != context.inputDimension) {
                    throw new IllegalArgumentException("Incorrect input dimension! other: " +
                            otherContext.inputDimension + " new: " + context.inputDimension);
                }

                if (otherContext.outputDimension != context.outputDimension) {
                    throw new IllegalArgumentException("Incorrect output dimension! other: " +
                            otherContext.outputDimension + " new: " + context.outputDimension);
                }

            });
            contexts.add(context);
            builder.addInput(context.sourceShape, context.targetShape, inputBuilder);
        }

        private TransferContext toTransferContext() {
            return contexts.stream().reduce((tc1, tc2) ->
                    new TransferContext(
                            elemsum(tc1.sourceShape, tc2.sourceShape),
                            elemsum(tc1.targetShape, tc2.targetShape),
                            tc1.inputDimension,
                            tc2.outputDimension))
                    .orElseThrow(() -> new IllegalStateException("Must have at least one context!"));
        }

        private static long[] elemsum(long[] a, long[] b) {
            return IntStream.range(0, a.length).mapToLong(i -> a[i] + b[i]).toArray();
        }

    }

    private static final class ParamPair {
        private final String layerName;
        private final Map<String, INDArray> source;
        private final Map<String, INDArray> target;

        private ParamPair(
                String layerName,
                Map<String, INDArray> source,
                Map<String, INDArray> target) {
            this.layerName = layerName;
            this.source = source;
            this.target = target;
        }
    }

    public ParameterTransfer(ComputationGraph sourceCompGraph) {
        this(sourceCompGraph, str -> Optional.empty());
    }

    public ParameterTransfer(Function<String, ComputationGraph> sourceCompGraph) {
        this(sourceCompGraph, str -> Optional.empty());
    }

    public ParameterTransfer(ComputationGraph sourceCompGraph, Function<String, Optional<Function<Integer, Comparator<Integer>>>> compFactory) {
        this.sourceCompGraph = sourceCompGraph;
        this.compFactory = compFactory;
        this.registry = new TransferRegistry();
    }

    public ParameterTransfer(Function<String, ComputationGraph> sourceCompGraph, Function<String, Optional<Function<Integer, Comparator<Integer>>>> compFactory) {
        this.sourceCompGraph = sourceCompGraph.apply("dummy");
        this.compFactory = compFactory;
        this.registry = new TransferRegistry();
    }

    /**
     * Mutate the current graph into a new {@link ComputationGraphConfiguration} by transferring parameters
     *
     * @param targetCompGraph new computation graph to which weights shall be transferred
     * @return A {@link ComputationGraph} initialized with parameters from the existing graph
     */
    public ComputationGraph transferWeightsTo(ComputationGraph targetCompGraph) {

        final int[] topologicalOrder = targetCompGraph.topologicalSortOrder();
        final GraphVertex[] vertices = targetCompGraph.getVertices();
        //set params from orig graph as necessary to new graph
        final Set<TransferTask.ListBuilder> builders = new LinkedHashSet<>();
        for (int vertexIndex : topologicalOrder) {
            builders.add(transferParameters(targetCompGraph, vertices[vertexIndex].getVertexName()));
        }
        final List<MergeTransferBuffer> mergeTransferBuffers = mergeTasks.values().stream().map(mc -> mc.builder.build()).collect(Collectors.toList());
        builders.stream().map(TransferTask.ListBuilder::build).forEachOrdered(TransferTask::execute);
        mergeTransferBuffers.forEach(MergeTransferBuffer::transferBufferedIndexes);
        registry.commit();
        return targetCompGraph;
    }

    private TransferTask.ListBuilder transferParameters(
            ComputationGraph targetCompGraph,
            String layerName) {
        Optional<ParamPair> paramPairOpt = getParams(layerName, sourceCompGraph, targetCompGraph);
        return paramPairOpt.map(paramPair -> {

            //System.out.println("Transfer start at " + layerName + " nInTransferred: " + nInTransferredFromPreviousNoutLayers);

            // Basically means "!doesSizeChangeTransfer"
            if (paramPair.source.containsKey(W)) {
                final TransferContext transferContext = new TransferContext(
                        paramPair.source.get(W).shape(),
                        paramPair.target.get(W).shape(),
                        inputDim(paramPair.source),
                        outputDim(paramPair.source));

                ////System.out.println("\t source " + Arrays.toString(transferContext.sourceShape) + " target " + Arrays.toString(transferContext.targetShape));
                TransferTask.ListBuilder taskBuilder = initTransfer(transferContext, paramPair);

                final Graph<String> traverseDependent = TraverseBuilder.forwards(targetCompGraph)
                        .andTraverseCondition(vertex -> !(targetCompGraph.getVertex(vertex) instanceof MergeVertex))
                        .allowRevisit()
                        .build();

                taskBuilder.addDependentTask(traverseGraph(
                        layerName,
                        targetCompGraph,
                        traverseDependent,
                        transferContext));

                return taskBuilder;
            } else {
                transferAllParameters(layerName, paramPair);
                return NoTransferTask.builder();
            }
        }).orElseGet(() -> {
            // Set weights to ID mapping (if possible) for new layers
            // TODO This decision does not really belong to this class!
            findLayerVertex(layerName, targetCompGraph)
                    .map(vertex -> vertex.paramTable(false))
                    .filter(parMap -> parMap.containsKey(W))
                    .map(parMap -> parMap.get(W))
                    .ifPresent(ParameterTransfer::setIdentityMapping);

            return NoTransferTask.builder();
        });
    }

    private TransferTask.ListBuilder initTransfer(
            TransferContext transferContext,
            ParamPair paramPair) {

        final SingleTransferTask.Builder firstTaskBuilder = SingleTransferTask.builder();
        if (nInTransferredFromPreviousNoutLayers.contains(paramPair.layerName)) {
            firstTaskBuilder
                    .maskDim(transferContext.inputDimension);
        }

        firstTaskBuilder
                .source(SingleTransferTask.IndMapping.builder()
                        .entry(registry.register(paramPair.source.get(W), paramPair.layerName + "_source_W"))
                        .build())
                .target(SingleTransferTask.IndMapping.builder()
                        .entry(registry.register(paramPair.target.get(W), paramPair.layerName + "_target_W"))
                        .build());

        compFactory.apply(paramPair.layerName)
                .ifPresent(firstTaskBuilder::compFactory);

        if (paramPair.source.containsKey(DefaultParamInitializer.BIAS_KEY)) {
            firstTaskBuilder
                    .addDependentTask(createDependentTask(
                            registry.register(paramPair.source.get(DefaultParamInitializer.BIAS_KEY), paramPair.layerName + "_source_b"),
                            registry.register(paramPair.target.get(DefaultParamInitializer.BIAS_KEY), paramPair.layerName + "_target_b"),
                            dim -> 1, // Always 1 for bias!
                            transferContext.inputDimension, 2, 3, 4)
                    );
        }
        if (paramPair.source.containsKey(CenterLossParamInitializer.CENTER_KEY)) {
            firstTaskBuilder
                    .addDependentTask(
                            createDependentTask(
                                    registry.register(paramPair.source.get(CenterLossParamInitializer.CENTER_KEY), paramPair.layerName + "_source_cl"),
                                    registry.register(paramPair.target.get(CenterLossParamInitializer.CENTER_KEY), paramPair.layerName + "_target_cl"),
                                    dim -> 1, // Always 1 for centerloss!
                                    2, 3, 4));
        }
        return firstTaskBuilder;
    }

    private TransferTask.ListBuilder traverseGraph(
            String inputVertex,
            ComputationGraph targetCompGraph,
            Graph<String> graph,
            TransferContext transferContext) {
        return graph.children(inputVertex).map(vertex -> {
            //System.out.println("Got " + vertex);
            final TransferTask.ListBuilder builder = new DependentTaskBuilder();

            // Add parameters (if any and if size allows for transfer) as dependent tasks
            getParams(vertex, sourceCompGraph, targetCompGraph)
                    .filter(paramPair -> canTransferNoutToNin(transferContext, paramPair))
                    .ifPresent(paramPair -> builder.addDependentTask(
                            addTasksFor(transferContext, paramPair)));

            // If we hit a mergevertex we must stop until we have dependent tasks for all of its inputs,
            // then we proceed.
            if (targetCompGraph.getVertex(vertex) instanceof MergeVertex) {
                //System.out.println("\t" + inputVertex + " hit a mergevertex " + vertex + "! contexts: " + mergeTasks);
                MergeContext mergeContext = mergeTasks.computeIfAbsent(vertex,
                        name -> new MergeContext());
                // How do we know that this happens in the right order? We trust the ComputationGraphs topological
                // order to be consistent with the order inputs to merge vertex are stacked.
                mergeContext.add(transferContext, builder);
                // Check if we have all inputs, then proceed further down the graph as we now have the right
                // source and target shapes to pass the canTransferNoutToNin check above
                if (targetCompGraph.getVertex(vertex).getInputVertices().length == mergeContext.contexts.size()) {
                    //System.out.println("\tProceed with mergevertex!");
                    mergeContext.builder.addDependentTask(traverseGraph(
                            vertex,
                            targetCompGraph,
                            graph,
                            mergeContext.toTransferContext()));
                }
            }
            return builder;
        })
                .reduce(TransferTask.ListBuilder::addDependentTask)
                .orElse(NoTransferTask.builder());
    }

    private TransferTask.ListBuilder addTasksFor(
            TransferContext transferContext,
            ParamPair paramPair) {

        final TransferTask.ListBuilder taskBuilder = new DependentTaskBuilder();
        if (!nInTransferredFromPreviousNoutLayers.add(paramPair.layerName)) {
            return taskBuilder;
        }
        // boolean[] first = {true};
        for (String parKey : paramPair.source.keySet()) {
            outputToInputDimMapping(parKey, paramPair.source.get(parKey).shape().length).ifPresent(dimMapper -> {

                final INDArray sourceParam = paramPair.source.get(parKey);
                final INDArray targetParam = paramPair.target.get(parKey);

//                if (first[0]) {
//                    System.out.println("\tTransfer dependent output: " + paramPair.layerName + " source " + Arrays.toString(sourceParam.shape()) + " target: " + Arrays.toString(targetParam.shape()));
//                    first[0] = false;
//                }

                if (sourceParam.size(0) != targetParam.size(0)
                        || sourceParam.size(1) != targetParam.size(1)) {
                    taskBuilder.addDependentTask(
                            createDependentTask(
                                    registry.register(sourceParam, paramPair.layerName + "_source_" + parKey),
                                    registry.register(targetParam, paramPair.layerName + "_target_" + parKey),
                                    dimMapper,
                                    // If the root task has changed inputs we don't want to transfer that
                                    transferContext.inputDimension, 2, 3, 4));
                }
            });
        }
        return taskBuilder;
    }

    private void transferAllParameters(String layerName,
                                       ParamPair paramPair) {

        // TODO Which key is most important?
        paramPair.source.keySet().stream().sorted().findFirst().ifPresent(startKey -> {
            final SingleTransferTask.Builder firstTaskBuilder = SingleTransferTask.builder()
                    .source(SingleTransferTask.IndMapping.builder()
                            .entry(registry.register(paramPair.source.get(startKey), layerName + "_source_" + startKey))
                            .build())
                    .target(SingleTransferTask.IndMapping.builder()
                            .entry(registry.register(paramPair.target.get(startKey), layerName + "_target_" + startKey))
                            .build());

            compFactory.apply(layerName)
                    .ifPresent(firstTaskBuilder::compFactory);
            paramPair.source.keySet().stream().filter(par -> !par.equals(startKey)).forEach(parKey ->
                    firstTaskBuilder
                            .addDependentTask(SingleTransferTask.builder()
                                    .source(SingleTransferTask.IndMapping.builder()
                                            .entry(registry.register(paramPair.source.get(parKey), layerName + "_source_" + parKey))
                                            .build())
                                    .target(SingleTransferTask.IndMapping.builder()
                                            .entry(registry.register(paramPair.target.get(parKey), layerName + "_target_" + parKey))
                                            .build()))
            );
            firstTaskBuilder.build().execute();
        });
    }

    private Optional<GraphVertex> findLayerVertex(String name, ComputationGraph graph) {
        return Optional.ofNullable(graph.getVertex(name))
                .filter(GraphVertex::hasLayer)
                .filter(vertex -> vertex.getLayer().numParams() > 0);
    }

    private boolean canTransferNoutToNin(TransferContext tc, ParamPair paramPair) {
        if (!paramPair.source.containsKey(W)) {
            return false;
        }

        final int inputDim = inputDim(paramPair.source);
//        //System.out.println("\t Test can transfer. Sources: " + tc.sourceShape[tc.outputDimension] + " vs " +
//                paramPair.source.get(W).size(inputDim) + " targets: " + tc.targetShape[tc.outputDimension] + " vs " +
//                paramPair.target.get(W).size(inputDim));
        return tc.sourceShape[tc.outputDimension] == paramPair.source.get(W).size(inputDim)
                && tc.targetShape[tc.outputDimension] == paramPair.target.get(W).size(inputDim);
    }

    private Optional<ParamPair> getParams(
            String layerName,
            ComputationGraph sourceGraph,
            ComputationGraph targetGraph) {

        final Optional<GraphVertex> sourceVertexMaybe = findLayerVertex(layerName, sourceGraph);
        final Optional<GraphVertex> targetVertexMaybe = findLayerVertex(layerName, targetGraph);

        if (sourceVertexMaybe.isPresent() && targetVertexMaybe.isPresent()) {
            final GraphVertex sourceVertex = sourceVertexMaybe.get();
            final GraphVertex targetVertex = targetVertexMaybe.get();

            return Optional.of(new ParamPair(
                    layerName,
                    sourceVertex.paramTable(false),
                    targetVertex.paramTable(false)));
        }
        return Optional.empty();
    }

    private static void setIdentityMapping(INDArray weights) {
        if (weights.size(0) != weights.size(1)) {
            // ID mapping not possible?
            return;
        }

        if (weights.shape().length == 2) {
            weights.assign(Nd4j.eye(weights.size(0)));
        } else if (weights.shape().length == 4 && weights.size(2) % 2 == 1 && weights.size(3) % 2 == 1) {
            weights.assign(Nd4j.zeros(weights.shape()));
            final long centerH = weights.size(2) / 2;
            final long centerW = weights.size(3) / 2;
            for (int i = 0; i < weights.size(0); i++) {
                weights.put(new INDArrayIndex[]{NDArrayIndex.point(i), NDArrayIndex.point(i), NDArrayIndex.point(centerH), NDArrayIndex.point(centerW)},
                        Nd4j.ones(1));
            }
        }
    }

    private static TransferTask.ListBuilder createDependentTask(
            TransferRegistry.ArrayEntry source,
            TransferRegistry.ArrayEntry target,
            IntUnaryOperator dimMapper,
            int... maskDims
    ) {
        return IntStream.of(maskDims).boxed().reduce(SingleTransferTask.builder()
                        .source(SingleTransferTask.IndMapping.builder()
                                .entry(source)
                                .dimensionMapper(dimMapper)
                                .build())
                        .target(SingleTransferTask.IndMapping.builder()
                                .entry(target)
                                .dimensionMapper(dimMapper)
                                .build()),
                SingleTransferTask.Builder::maskDim,
                (b1, b2) -> b1);
    }

    /**
     * Describes how to map output dimension from a previous layer to a input dimension of a next layer for different parameters
     *
     * @param paramName The weight key name
     * @return The mapping. empty means weight key shall not be mapped
     */
    private Optional<IntUnaryOperator> outputToInputDimMapping(String paramName, int rank) {
        switch (paramName) {
            case W:
                switch (rank) {
                    case 2:
                        return Optional.of(dim -> 0); // dim 0 is input to dense layers
                    case 4:
                        return Optional.of(dim -> 1); // dim 1 is input to conv layers
                    default:
                        throw new UnsupportedOperationException("Not supported yet: " + rank);
                }
            case (CenterLossParamInitializer.CENTER_KEY):
                return Optional.of(dim -> 1); // dim 1 is input to center loss
            case (DefaultParamInitializer.BIAS_KEY):
            case (BatchNormalizationParamInitializer.BETA):
            case (BatchNormalizationParamInitializer.GAMMA):
            case (BatchNormalizationParamInitializer.GLOBAL_MEAN):
            case (BatchNormalizationParamInitializer.GLOBAL_VAR):
                return Optional.empty();
            default:
                throw new UnsupportedOperationException("Param type not supported: " + paramName);
        }
    }

    private static int inputDim(Map<String, INDArray> params) {
        if (params.containsKey(W)) {
            // 1 for conv, 0 for dense ?? for recurrent
            return params.get(W).shape().length == 4 ? 1 : 0;
        }
        return 1;
    }

    private static int outputDim(Map<String, INDArray> params) {
        return inputDim(params) == 0 ? 1 : 0;
    }
}
