package ampcontrol.model.training.model.evolve.crossover.graph;

import ampcontrol.model.training.model.evolve.mutate.util.ForwardOf;
import ampcontrol.model.training.model.evolve.mutate.util.GraphBuilderUtil;
import ampcontrol.model.training.model.evolve.mutate.util.Traverse;
import ampcontrol.model.training.model.evolve.mutate.util.TraverseBuilder;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration.GraphBuilder;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.GraphVertex;
import org.deeplearning4j.nn.conf.graph.LayerVertex;
import org.jetbrains.annotations.NotNull;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * Representation of a crossover point between two {@link VertexData}s. Can be executed in order to create a new
 * {@link GraphInfo} with the result of the crossover.
 *
 * @author Christian Skärby
 */
class CrossoverPoint {

    private static final Logger log = LoggerFactory.getLogger(CrossoverPoint.class);

    private final VertexData bottom;
    private final VertexData top;
    private final double distance;

    CrossoverPoint(VertexData bottom, VertexData top) {
        this.bottom = bottom;
        this.top = top;
        distance = (bottom.location() - top.location()) / 2;
        //System.out.println("valid point " + bottom.name() + ", " + top.name() + " dist: " + distance);
    }

    double distance() {
        return distance;
    }

    VertexData bottom() {
        return bottom;
    }

    VertexData top() {
        return top;
    }

    /**
     * Execute the crossover between the two vertices
     * @return a {@link GraphInfo} which apart from the new {@link ComputationGraphConfiguration.GraphBuilder}
     */
    GraphInfo execute() {
        log.info("Do crossover between " + bottom.name() + " and " + top.name() + " with distance " + distance);
        //System.out.println("Do crossover between " + bottom.name() + " and " + top.name() + " with distance " + distance);

        final long oldNOut = GraphBuilderUtil.getOutputSize(bottom.name(), bottom.builder());
        final long oldNin = GraphBuilderUtil.getInputSize(top.name(), top.builder());

        final GraphBuilder builder = initBuilder();
        final Set<String> bottomVertices = new LinkedHashSet<>(builder.getVertices().keySet());
        final Map<String, String> topVerticesNameMapping = addTop(builder);

        alignNoutNin(builder,
                topVerticesNameMapping,
                oldNOut,
                oldNin);

        //System.out.println("new graph: " + builder.getVertexInputs() + " inputs " + builder.getNetworkInputs());

        // What happens here? The user needs to be able to query the result for which vertices are from which "input"
        // info.
        // To satisfy this requirement, bottom is simply mapped to a "filter" which only lets "bottomVertices" through it.
        // For top, things are a little bit more complicated as the vertices might have been renamed to avoid the same name
        // being reused. We did however create the mapping already so we just use it to map names from the "input" builder
        // (i.e top.builder) to the names used in the "output" builder (i.e builder).

        final Map<GraphInfo, GraphInfo> infoMap = new HashMap<>();
        infoMap.put(bottom.info(), new GraphInfo.NameMap(bottom.info(), name -> bottomVertices.contains(name) ? name : null));
        infoMap.put(top.info(), new GraphInfo.NameMap(top.info(), topVerticesNameMapping::get));
        return new GraphInfo.Result(builder, infoMap);
    }

    @NotNull
    private GraphBuilder initBuilder() {
        final ComputationGraphConfiguration conf = bottom.builder().build();
        final GraphBuilder builder = new GraphBuilder(
                conf, new NeuralNetConfiguration.Builder(conf.getDefaultConfiguration()));

        new Traverse<>(new ForwardOf(builder)).children(bottom.name())
                .forEach(vertex -> builder.removeVertex(vertex, true));
        return builder;
    }

    @NotNull
    private Map<String, String> addTop(GraphBuilder builder) {
        // Need to access names which are already added to the builder below, so first create the mapping
        // between names in top and a name which is unique in builder.
        final Map<String, String> topVerticesNameMapping = TraverseBuilder.forwards(top.builder())
                .traverseCondition(vertex -> true)
                .build().children(top.name())
                .collect(Collectors.toMap(
                        vertex -> vertex,
                        vertex -> checkName(vertex, builder)
                ));
        topVerticesNameMapping.put(top.name(), checkName(top.name(), builder));
        //System.out.println("topVerts: " + topVerticesNameMapping.keySet());

        builder.addVertex(topVerticesNameMapping.get(top.name()),
                renameVertexIfLayer(top.builder().getVertices().get(top.name()),  topVerticesNameMapping.get(top.name())),
                bottom.name());

        new Traverse<>(new ForwardOf(top.builder())).children(top.name())
                .forEach(vertex -> builder.addVertex(
                        topVerticesNameMapping.get(vertex),
                        renameVertexIfLayer(top.builder().getVertices().get(vertex), topVerticesNameMapping.get(vertex)),
                        top.builder().getVertexInputs().get(vertex)
                                .stream()
                                .map(topVerticesNameMapping::get)
                                .collect(Collectors.toList())
                                .toArray(new String[0])));
        builder.setOutputs(top.builder().getNetworkOutputs().toArray(new String[0]));
        return topVerticesNameMapping;
    }

    private static GraphVertex renameVertexIfLayer(GraphVertex vertex, String newName) {
        if(vertex instanceof LayerVertex) {
            final LayerVertex newVert = (LayerVertex)vertex.clone();
            newVert.getLayerConf().getLayer().setLayerName(newName);
            return  newVert;
        }
        return vertex;
    }

    private static String checkName(String wantedName, GraphBuilder builder) {
        return createUniqueVertexName(0, wantedName, wantedName, builder);
    }

    private static String createUniqueVertexName(int cnt, String wantedName, String toCheck, GraphBuilder builder) {

        return builder.getVertices().keySet().stream()
                .filter(toCheck::equals)
                .map(name -> createUniqueVertexName(cnt + 1, wantedName, wantedName + "_" + cnt, builder))
                .findAny()
                .orElse(toCheck);
    }

    private void alignNoutNin(
            GraphBuilder builder,
            Map<String, String> topVerticesNameMapping,
            long oldNOut,
            long oldNIn) {
        // Select the option which does not result in removal of weights
        if (oldNIn > oldNOut) {
            final long newNOut = oldNIn;
            TraverseBuilder.backwards(builder)
                    .enterCondition(vertex -> true)
                    .build().children(topVerticesNameMapping.get(top.name()))
                    .forEach(vertex ->
                            GraphBuilderUtil.asFeedforwardLayer(builder).apply(vertex)
                                    .ifPresent(layer -> {
                                        layer.setNOut(newNOut);
                                        if (GraphBuilderUtil.changeSizePropagates(builder).test(vertex)) {
                                            // Means nOut must be equal to nIn
                                            layer.setNIn(newNOut);
                                        }
                                    })
                    );
        } else {
            final long newNIn = oldNOut;
            TraverseBuilder.forwards(builder)
                    .enterCondition(vertex -> true)
                    .build().children(bottom.name())
                    .forEach(vertex ->
                            GraphBuilderUtil.asFeedforwardLayer(builder).apply(vertex)
                                    .ifPresent(layer -> {
                                        layer.setNIn(newNIn);
                                        if (GraphBuilderUtil.changeSizePropagates(builder).test(vertex)) {
                                            // Means nOut must be equal to nIn
                                            layer.setNOut(newNIn);
                                        }
                                    })
                    );
        }
    }
}
