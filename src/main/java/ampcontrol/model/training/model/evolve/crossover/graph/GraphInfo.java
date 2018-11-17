package ampcontrol.model.training.model.evolve.crossover.graph;

import lombok.AllArgsConstructor;
import lombok.Getter;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration.GraphBuilder;

import java.util.Map;
import java.util.Objects;
import java.util.function.Function;
import java.util.stream.Stream;

/**
 * Represents information about a {@link GraphBuilder} in a crossover context
 */
public interface GraphInfo {

    @Getter
    @AllArgsConstructor
    final class NameMapping {
        private final String oldName;
        private final String newName;

        NameMapping remap(Function<String, String> map) {
            return new NameMapping(oldName, map.apply(newName));
        }
    }
    
    final class Input implements GraphInfo {

        private final GraphBuilder builder;

        public Input(GraphBuilder builder) {
            this.builder = builder;
        }

        @Override
        public GraphBuilder builder() {
            return builder;
        }

        @Override
        public Stream<NameMapping> verticesFrom(GraphInfo info) {
            if (info != this) {
                throw new IllegalArgumentException("Incorrect builder!");
            }
            return this.builder.getVertices().keySet()
                    .stream()
                    .map(vertex -> new NameMapping(vertex,vertex));
        }
    }

    final class Result implements GraphInfo {

        private final GraphBuilder builder;
        private final Map<GraphInfo, GraphInfo> vertices;

        Result(GraphBuilder builder, Map<GraphInfo, GraphInfo> vertices) {
            this.builder = builder;
            this.vertices = vertices;
        }

        @Override
        public GraphBuilder builder() {
            return builder;
        }

        @Override
        public Stream<NameMapping> verticesFrom(GraphInfo info) {
            if (info == this) {
                return builder.getVertices().keySet()
                        .stream()
                        .map(vertex -> new NameMapping(vertex,vertex));
            }
            return vertices.get(info).verticesFrom(this);
        }
    }

    final class NameMap implements GraphInfo {

        private final GraphInfo info;
        private final Function<String, String> nameMap;

        NameMap(GraphInfo info, Function<String, String> nameMap) {
            this.info = info;
            this.nameMap = nameMap;
        }

        @Override
        public GraphBuilder builder() {
            return info.builder();
        }

        @Override
        public Stream<NameMapping> verticesFrom(GraphInfo info) {
            return info.verticesFrom(info)
                    .map(nameMapping -> nameMapping.remap(nameMap))
                    .filter(nameMapping -> Objects.nonNull(nameMapping.newName));
        }
    }


    GraphBuilder builder();

    Stream<NameMapping> verticesFrom(GraphInfo info);
}
