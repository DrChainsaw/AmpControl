package ampcontrol.model.training.model.naming;

/**
 * Adds a suffix String to the given {@link FileNamePolicy}
 *
 * @author Christian Skärby
 */
public class AddSuffix implements FileNamePolicy {

    private final String suffix;

    /**
     * Constructor
     * @param suffix Suffix to add
     */
    public AddSuffix(String suffix) {
        this.suffix = suffix;
    }

    @Override
    public String toFileName(String str) {
        return str + suffix;
    }
}
