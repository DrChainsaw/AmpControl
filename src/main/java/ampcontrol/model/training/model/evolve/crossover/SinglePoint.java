package ampcontrol.model.training.model.evolve.crossover;

import ampcontrol.model.training.model.evolve.mutate.util.ForwardOf;
import ampcontrol.model.training.model.evolve.mutate.util.GraphBuilderUtil;
import ampcontrol.model.training.model.evolve.mutate.util.Traverse;
import ampcontrol.model.training.model.evolve.mutate.util.TraverseBuilder;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration.GraphBuilder;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.ElementWiseVertex;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.jetbrains.annotations.NotNull;

import java.util.*;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Simple {@link Crossover} for {@link GraphBuilder}s. To keep things reasonably simple, only vertices which are the
 * single vertex connecting all vertices above and below them are eligible for crossover.
 * <br><br>
 * Assumes the input types have been set for the inputs.
 *
 * @author Christian Skärby
 */
public final class SinglePoint implements Crossover<SinglePoint.Info> {

    private final Supplier<PointSelection> selectionSupplier;

    public static class PointSelection {
        private final double relativeLocation;
        private final double distance;

        public PointSelection(double distance, double relativeLocation) {
            this.relativeLocation = relativeLocation;
            this.distance = distance;
        }

        public double getRelativeLocation() {
            return relativeLocation;
        }

        public double getDistance() {
            return distance;
        }
    }

    public interface Info {
        GraphBuilder builder();

        Stream<String> verticesFrom(Info info);
    }

    public final static class Input implements Info {

        private final GraphBuilder builder;

        public Input(GraphBuilder builder) {
            this.builder = builder;
        }

        @Override
        public GraphBuilder builder() {
            return builder;
        }

        @Override
        public Stream<String> verticesFrom(Info info) {
            if (info != this) {
                throw new IllegalArgumentException("Incorrect builder!");
            }
            return this.builder.getVertices().keySet().stream();
        }
    }

    public final static class Result implements Info {

        private final GraphBuilder builder;
        private final Map<Info, Info> vertices;

        public Result(GraphBuilder builder, Map<Info, Info> vertices) {
            this.builder = builder;
            this.vertices = vertices;
        }

        @Override
        public GraphBuilder builder() {
            return builder;
        }

        @Override
        public Stream<String> verticesFrom(Info info) {
            if (info == this) {
                return builder.getVertices().keySet().stream();
            }
            return vertices.get(info).verticesFrom(this);
        }
    }

    private final static class NameMap implements Info {

        private final Info info;
        private final Function<String, String> nameMap;

        private NameMap(Info info, Function<String, String> nameMap) {
            this.info = info;
            this.nameMap = nameMap;
        }

        @Override
        public GraphBuilder builder() {
            return info.builder();
        }

        @Override
        public Stream<String> verticesFrom(Info info) {
            return info.verticesFrom(info)
                    .map(nameMap)
                    .filter(Objects::nonNull);
        }
    }

    private static class VertexData {
        private final String name;
        private final Info info;
        private final double relativeLocation;

        private VertexData(String name, Info info) {
            this.name = name;
            this.info = info;
            this.relativeLocation = 1 - new Traverse<>(
                    vert -> !vert.equals(name),
                    new ForwardOf(builder()))
                    .children(name).count() / (double) builder().getVertices().size();
            //  System.out.println("Valid vert " + name + " lcc " + relativeLocation);
        }

        private InputType.Type type() {
            return builder().getLayerActivationTypes().get(name).getType();
        }

        private GraphBuilder builder() {
            return info.builder();
        }

        private double location() {
            return relativeLocation;
        }

        private Info info() {
            return info;
        }
    }

    private static class CrossoverPoint {

        private final VertexData bottom;
        private final VertexData top;
        private final double distance;

        private CrossoverPoint(VertexData bottom, VertexData top) {
            this.bottom = bottom;
            this.top = top;
            distance = Math.abs(bottom.location() - top.location());
            //System.out.println("valid point " + bottom.name + ", " + top.name + " dist: " + distance);
        }

        private double distance() {
            return distance;
        }

        private Info execute() {
            //System.out.println("Do crossover between " + bottom.name + " and " + top.name + " with distance " + distance);
            final long oldNOut = GraphBuilderUtil.getOutputSize(bottom.name, bottom.builder());
            final long oldNin = GraphBuilderUtil.getInputSize(top.name, top.builder());

            final GraphBuilder builder = initBuilder();
            final Set<String> bottomVertices = new LinkedHashSet<>(builder.getVertices().keySet());
            final Map<String, String> topVerticesNameMapping = addTop(builder);

            alignNoutNin(builder, oldNOut, oldNin);

            //System.out.println("new graph: " + builder.getVertexInputs());

            final Map<Info, Info> infoMap = new HashMap<>();
            infoMap.put(bottom.info(), new NameMap(bottom.info(), name -> bottomVertices.contains(name) ? name : null));
            infoMap.put(top.info(), new NameMap(top.info(), topVerticesNameMapping::get));
            return new Result(builder, infoMap);
        }

        @NotNull
        private GraphBuilder initBuilder() {
            final ComputationGraphConfiguration conf = bottom.builder().build();
            final GraphBuilder builder = new GraphBuilder(
                    conf, new NeuralNetConfiguration.Builder(conf.getDefaultConfiguration()));

            new Traverse<>(new ForwardOf(builder)).children(bottom.name)
                    .forEach(vertex -> builder.removeVertex(vertex, true));
            return builder;
        }

        @NotNull
        private Map<String, String> addTop(GraphBuilder builder) {
            // Need to access names which are already added to the builder below, so first create the mapping
            // between names in top and a name which is unique in builder.
            final Map<String, String> topVerticesNameMapping = TraverseBuilder.forwards(top.builder())
                    .traverseCondition(vertex -> true)
                    .build().children(top.name)
                    .collect(Collectors.toMap(
                            vertex -> vertex,
                            vertex -> checkName(vertex, builder)
                    ));
            topVerticesNameMapping.put(top.name, checkName(top.name, builder));
            //System.out.println("topVerts: " + topVerticesNameMapping.keySet());

            builder.addVertex(topVerticesNameMapping.get(top.name), top.builder().getVertices().get(top.name), bottom.name);

            new Traverse<>(new ForwardOf(top.builder())).children(top.name)
                    .forEach(vertex -> builder.addVertex(
                            topVerticesNameMapping.get(vertex),
                            top.builder().getVertices().get(vertex),
                            top.builder().getVertexInputs().get(vertex)
                                    .stream()
                                    .map(topVerticesNameMapping::get)
                                    .collect(Collectors.toList())
                                    .toArray(new String[0])));
            builder.setOutputs(top.builder().getNetworkOutputs().toArray(new String[0]));
            return topVerticesNameMapping;
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

        private void alignNoutNin(GraphBuilder builder, long oldNOut, long oldNIn) {
            // Select the option which does not result in removal of weights
            if(oldNIn > oldNOut) {
                final long newNOut = oldNIn;
                TraverseBuilder.backwards(builder)
                        .enterCondition(vertex -> true)
                        .build().children(top.name)
                        .forEach(vertex ->
                            GraphBuilderUtil.asFeedforwardLayer(builder).apply(vertex)
                                    .ifPresent(layer -> {
                                        layer.setNOut(newNOut);
                                        if(GraphBuilderUtil.changeSizePropagates(builder).test(vertex)) {
                                            // Means nOut must be equal to nIn
                                            layer.setNIn(newNOut);
                                        }
                                    })
                        );
            } else {
                final long newNIn = oldNOut;
                TraverseBuilder.forwards(builder)
                        .enterCondition(vertex -> true)
                        .build().children(bottom.name)
                        .forEach(vertex ->
                                GraphBuilderUtil.asFeedforwardLayer(builder).apply(vertex)
                                        .ifPresent(layer -> {
                                            layer.setNIn(newNIn);
                                            if(GraphBuilderUtil.changeSizePropagates(builder).test(vertex)) {
                                                // Means nOut must be equal to nIn
                                                layer.setNOut(newNIn);
                                            }
                                        })
                        );
            }
        }
    }

    public SinglePoint(Supplier<PointSelection> selectionSupplier) {
        this.selectionSupplier = selectionSupplier;
    }

    @Override
    public Info cross(
            Info bottom,
            Info top) {

        final PointSelection pointSelection = selectionSupplier.get();

        final Comparator<CrossoverPoint> byDistance = Comparator.comparingDouble(
                point -> Math.abs(pointSelection.getDistance() - point.distance()));
        final Comparator<CrossoverPoint> byBottomLocation = Comparator.comparingDouble(
                point -> Math.abs(pointSelection.getRelativeLocation() - point.bottom.relativeLocation));
        final Comparator<CrossoverPoint> byTiebreaker = Comparator.comparingInt(point -> (point.bottom.name + point.top.name).hashCode());

        return findValidVertices(bottom)
                .flatMap(bottomVertex -> findValidVertices(top)
                        .filter(topVertex -> topVertex.type() == bottomVertex.type())
                        .map(topVertex -> new CrossoverPoint(bottomVertex, topVertex)))
                .min(byDistance.thenComparing(byBottomLocation).thenComparing(byTiebreaker))
                .map(CrossoverPoint::execute)
                .orElse(bottom); // Not a single valid crossover point found
    }

    private static Stream<VertexData> findValidVertices(Info info) {
        return info.builder().getVertices().keySet().stream()
                .filter(vertex -> isValid(vertex, info.builder()))
                .map(vertex -> new VertexData(vertex, info));

    }

    private static boolean isValid(String vertex, GraphBuilder builder) {
        // Vertices could perhaps work, but would just be a waste as they will typically only be connected to one input
        if (
                builder.getVertices().get(vertex) instanceof ElementWiseVertex
                || builder.getVertices().get(vertex) instanceof MergeVertex
                || builder.getNetworkOutputs().contains(vertex)
                || builder.getNetworkInputs().contains(vertex)) {
            return false;
        }


        final List<String> flattened = new Traverse<>(new ForwardOf(builder)).children(vertex)
                .collect(Collectors.toList());
        flattened.add(vertex);
        //Vertex is not inside any type of skip connection if we find all inputs to each found vertex in the flattened graph
        return flattened.stream()
                .filter(vert -> !vert.equals(vertex))

                .allMatch(vert -> flattened.containsAll(builder.getVertexInputs().get(vert)));
    }
}
