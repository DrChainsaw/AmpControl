package ampcontrol.model.training.model.evolve.state;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.File;
import java.io.IOException;
import java.util.Collection;
import java.util.LinkedHashSet;
import java.util.Set;

/**
 * A set which may persist in a file.
 *
 * @param <T>
 * @author Christian Skärby
 */
public class PersistentSet<T> {

    private final File file;
    private final Set<T> set;

    public PersistentSet(
            String fileName,
            Collection<T> set) throws IOException {
        this.file = new File(fileName);
        this.set = new LinkedHashSet<>(set);
        if (file.exists()) {
            TypeReference ref = new TypeReference<Set<T>>() {};
            this.set.addAll(new ObjectMapper().readValue(file, ref));
        }
    }

    /**
     * Returns the {@link Set}. Any changes made are backed by the set and may be persisted.
     *
     * @return the {@link Set}
     */
    public Set<T> get() {
        return set;
    }

    /**
     * Persist the set in a file
     *
     * @throws IOException
     */
    public void save() throws IOException {
        file.getParentFile().mkdirs();
        new ObjectMapper().writeValue(file, set);
    }
}
