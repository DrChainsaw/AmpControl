package ampcontrol.model.training.model.naming;

/**
 * Adds a prefix to the output from another {@link FileNamePolicy}. If no {@link FileNamePolicy} is provided the prefix
 * will be added to input String as is
 *
 * @author Christian Skärby
 */
public class AddPrefix implements FileNamePolicy {

    private final String prefix;

    /**
     * Constructor
     * @param prefix prefix to add
     */
    public AddPrefix(String prefix) {
        this.prefix = prefix;
    }

    @Override
    public String toFileName(String str) {
        return prefix + str;
    }
}
