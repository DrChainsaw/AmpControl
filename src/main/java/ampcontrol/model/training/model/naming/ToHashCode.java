package ampcontrol.model.training.model.naming;

/**
 * Returns hashCode of the input string. "Cheap" way to compress long strings to prevent that filesystem can't handle
 * the name. Note that this is not 100% safe in theory since infinitely many strings have the same hashcode. In practice
 * it seems to work ok.
 *
 * @author Christian Skärby
 */
public class ToHashCode implements FileNamePolicy {

    @Override
    public String toFileName(String str) {
        return String.valueOf(str.hashCode());
    }
}
