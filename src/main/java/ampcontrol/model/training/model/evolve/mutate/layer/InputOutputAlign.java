package ampcontrol.model.training.model.evolve.mutate.layer;

import ampcontrol.model.training.model.evolve.mutate.util.*;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.graph.ElementWiseVertex;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer;
import org.jetbrains.annotations.NotNull;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static ampcontrol.model.training.model.evolve.mutate.util.GraphBuilderUtil.isSizeChangePossible;

/**
 * Aligns output sizes to input sizes or vice versa depending on what prevents neurons from being removed.
 *
 * @author Christian Skärby
 */
public class InputOutputAlign {
    private static final Logger log = LoggerFactory.getLogger(InputOutputAlign.class);

    private final ComputationGraphConfiguration.GraphBuilder graphBuilder;
    private final List<String> outputNames;
    private final List<String> inputNames;
    private final long nOut;
    private final long nIn;

    public InputOutputAlign(
            ComputationGraphConfiguration.GraphBuilder graphBuilder,
            List<String> outputNames,
            List<String> inputNames,
            long nOut,
            long nIn) {
        this.graphBuilder = graphBuilder;
        this.outputNames = outputNames;
        this.inputNames = inputNames;
        this.nOut = nOut;
        this.nIn = nIn;
    }

    public void invoke() {
        // Not possible to change network inputs (e.g. image size)
        final boolean isAnyLayerTouchingNetworkInput =
        //graphBuilder.getNetworkInputs().stream().anyMatch(inputNames::contains);
        isAnyLayerTouchingNetworkInput(graphBuilder, inputNames);

        //System.out.println("Touches input " + isAnyLayerTouchingNetworkInput);

        // Do the change which adds neurons rather than the one which removes them
        // What about if nIn == nOut? Can't do early return it seems as this is no guarantee
        // that the below is not needed. Example when it is not involve pooling layers
        // and merge vertices

        // Why is not NoutMutation used for this? Because NoutMutation actually does something quite different:
        // Firstly it assumes that the graph is consistent to begin with w.r.t nOut and nIns (not the case here).
        // Secondly, it only needs to propagate the nOut forwards to subsequent layers nIn, only skipping
        // backwards when it encounters an ElementWiseVertex. Here we need to go backwards and fix the nOuts of
        // previous layers which are changed either because they are to be connected with the removed layers
        // outputs or because one of the paths in a fork was just removed.
        if (nIn > nOut || isAnyLayerTouchingNetworkInput) {
            //System.out.println("change nIn " + nIn);
            setNinOfOutputsToNoutSize(graphBuilder, outputNames);
        } else {
            //System.out.println("change nout : " + nOut);
            changeNoutOfInputs(graphBuilder, inputNames, nOut);
        }
    }

    private static boolean isAnyLayerTouchingNetworkInput(ComputationGraphConfiguration.GraphBuilder graphBuilder, List<String> inputNames) {
        final Graph<String> traverseBackwards = TraverseBuilder.backwards(graphBuilder)
                .enterCondition(vertex -> !isSizeChangePossible(graphBuilder.getVertices().get(vertex)))
                .allowRevisit()
                .build();

        return Stream.concat(inputNames.stream(), inputNames.stream().flatMap(traverseBackwards::children))
                .anyMatch(vertex -> graphBuilder.getNetworkInputs().contains(vertex));
    }

    private static void changeNoutOfInputs(ComputationGraphConfiguration.GraphBuilder graphBuilder, Collection<String> inputNames, long nOut) {

        //System.out.println("inputnames: " + inputNames);
        // What we want here is to traverse in topological order really. Just so happens to be so that inputNames
        // is always in reverse topological order since this is how it is constructed?
        final List<String> names = new ArrayList<>(inputNames);
        Collections.reverse(names);
        //System.out.println("reverse: " + names);

        final SizeVisitor sizeRegistry = createSizeVisitor(graphBuilder, nOut);
        inputNames.forEach(vertex -> sizeRegistry.set(vertex, nOut));

        final Set<String> changedLayers = new LinkedHashSet<>();
        toLayerStream(
                TraverseBuilder.backwards(graphBuilder)
                        .enterCondition(GraphBuilderUtil.changeSizePropagates(graphBuilder))
                        .enterListener(sizeRegistry::visit)
                        .build(),
                graphBuilder,
                names)
                .peek(layer -> log.info("Change nOut of layer " + layer.getLayerName() + " from " + layer.getNOut() + " to " + sizeRegistry.getSize(layer.getLayerName())))
                .forEachOrdered(layer -> {
                    final long thisNout = sizeRegistry.getSize(layer.getLayerName());
                    //System.out.println("change nOut of vertex " + layer.getLayerName() + " from " + layer.getNOut() + " to " + thisNout);
                    layer.setNOut(thisNout);
                    if (!isSizeChangePossible(layer)) {
                        layer.setNIn(thisNout);
                    } else {
                        changedLayers.add(layer.getLayerName());
                    }
                });

        // Set Nin of layers which have changed and are not part of inputNames
        final Graph<String> forward = TraverseBuilder.forwards(graphBuilder)
                .allowRevisit()
                .build();
        final Set<String> needToChangeNin = changedLayers.stream()
                .flatMap(forward::children)
                // We only want to process feedforward layers.
                .filter(vertex -> GraphBuilderUtil.asFeedforwardLayer(graphBuilder).apply(vertex).isPresent())
                .collect(Collectors.toSet());
        //System.out.println("Change nIns after changing nOuts " + needToChangeNin + " changed layers " + changedLayers);
        setNinOfOutputsToNoutSize(graphBuilder, needToChangeNin);
    }

    @NotNull
    private static SizeVisitor createSizeVisitor(ComputationGraphConfiguration.GraphBuilder graphBuilder, long nOut) {
        final Graph<String> backward = new BackwardOf(graphBuilder);
        final Graph<String> traverseMerges = Traverse.leaves(
                vert -> graphBuilder.getVertices().get(vert) instanceof MergeVertex, backward);

        return new SizeVisitor(
                // Whats going on here? First, we want to traverse through MergeVertices to give fair sharing
                // between the inputs to them given that they can have different sizes and be of different numbers.
                // However, in case of an ElementWiseVertex, we don't want to do this as this would set the inputs
                // to the inputs to the MergeVertex to the same nOut as the nOut of the ElementWiseVertex -> error!
                vertex -> (graphBuilder.getVertices().get(vertex) instanceof ElementWiseVertex) ? backward.children(vertex) : traverseMerges.children(vertex),
                graphBuilder,
                nOut,
                (layerSize, size) -> Math.max(1, size));
    }

    private static void setNinOfOutputsToNoutSize(ComputationGraphConfiguration.GraphBuilder graphBuilder, Collection<String> outputNames) {
        //System.out.println("output names: " + outputNames);
        log.info("Set NIn of outputs " + outputNames);


        final Graph<String> traverseInputs = GraphBuilderUtil.inputSizeTravere(graphBuilder)
                .traverseCondition(vertex -> !GraphBuilderUtil.asFeedforwardLayer(graphBuilder).apply(vertex).isPresent())
                .allowRevisit()
                .build();
        toLayerStream(
                TraverseBuilder.forwards(graphBuilder)
                        .enterCondition(GraphBuilderUtil.changeSizePropagates(graphBuilder))
                        .build(),
                graphBuilder,
                outputNames)
                .forEachOrdered(layer -> {
                    final long nInToUse = traverseInputs.children(layer.getLayerName())
                            .distinct()
                            //.peek(vertex -> System.out.println("visit " + vertex ))
                            .mapToLong(vertex -> GraphBuilderUtil.asFeedforwardLayer(graphBuilder).apply(vertex)
                                    .map(FeedForwardLayer::getNOut)
                                    .orElseGet(() -> graphBuilder.getNetworkInputs().contains(vertex)
                                            ? graphBuilder.getNetworkInputTypes().get(graphBuilder.getNetworkInputs().indexOf(vertex)).getShape(false)[0]
                                            : 0L))
                            //.peek(size -> System.out.println("size: " + size))
                            .sum();
                    log.info("Change nIn of layer " + layer.getLayerName() + " from " + layer.getNIn() + " to " + nInToUse);
                    //System.out.println("change nIn of vertex " + layer.getLayerName() + " from " + layer.getNIn() + " to " + nInToUse);
                    layer.setNIn(nInToUse);
                    if (!isSizeChangePossible(layer)) {
                        layer.setNOut(nInToUse);
                    }
                });
    }

    private static Stream<FeedForwardLayer> toLayerStream(
            Graph<String> graph,
            ComputationGraphConfiguration.GraphBuilder graphBuilder,
            Collection<String> names) {
        return
                Stream.concat(names.stream(), names.stream().flatMap(graph::children))
                        .map(GraphBuilderUtil.asFeedforwardLayer(graphBuilder))
                        .filter(Optional::isPresent)
                        .map(Optional::get);
    }
}
