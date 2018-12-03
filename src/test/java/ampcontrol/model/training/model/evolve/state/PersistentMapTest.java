package ampcontrol.model.training.model.evolve.state;

import org.junit.Test;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;

/**
 * Test cases for {@link PersistentMap}
 *
 * @author Christian Skärby
 */
public class PersistentMapTest {

    /**
     * Test persisting a map and then loading it
     */
    @Test
    public void persist() {
        final Path baseDir = Paths.get("src", "test", "resources", "PersistentMapTest");
        final String fileName = Paths.get(baseDir.toString(), "get.json").toString();

        try {
            final PersistentMap<String, Integer> expected = new PersistentMap<>(fileName, Stream.of("a", "test", "set").collect(Collectors.toMap(
                    str -> str,
                    String::hashCode
            )));
            assertFalse("Map shall not be empty!", expected.get().isEmpty());
            expected.save();
            final PersistentMap<String, Integer> actual = new PersistentMap<>(fileName, Collections.emptyMap());
            assertEquals("Set did not persist!", expected.get(), actual.get());
            new File(fileName).delete();
            Files.delete(baseDir);
        } catch (IOException e) {
            throw new IllegalStateException();
        }

    }
}