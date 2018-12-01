package ampcontrol.model.training.model.evolve.state;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.File;
import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * A map which may persist in a file.
 *
 * @param <K> Type of keys
 * @param <V> Type of values
 * @author Christian Skärby
 */
public class PersistentMap<K,V> {

    private final File file;
    private final Map<K,V> map;

    public PersistentMap(
            String fileName,
            Map<K,V> map) throws IOException {
        this.file = new File(fileName);
        this.map = new LinkedHashMap<>(map);
        if (file.exists()) {
            TypeReference ref = new TypeReference<Map<K,V>>() {};
            this.map.putAll(new ObjectMapper().readValue(file, ref));
        }
    }

    /**
     * Returns the {@link Map}. Any changes made are backed by the map and may be persisted.
     *
     * @return the {@link Map}
     */
    public Map<K,V> get() {
        return map;
    }

    /**
     * Persist the map in a file
     *
     * @throws IOException
     */
    public void save() throws IOException {
        file.getParentFile().mkdirs();
        new ObjectMapper().writeValue(file, map);
    }
}
