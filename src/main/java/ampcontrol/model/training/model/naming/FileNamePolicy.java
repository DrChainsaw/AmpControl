package ampcontrol.model.training.model.naming;

/**
 * Turns a String to a suitable filename, typically by adding extensions or a root path.
 *
 * @author Christian Skärby
 */
public interface FileNamePolicy {

    /**
     * Transparent policy
     */
    FileNamePolicy identity = str -> str;

    /**
     * Returns hashCode of the input string. "Cheap" way to compress long strings to prevent that filesystem can't handle
     * the name. Note that this is not 100% safe in theory since infinitely many strings have the same hashcode. In practice
     * it seems to work ok.
     */
    FileNamePolicy toHashCode = str -> String.valueOf(str.hashCode());


    /**
     * Translate the given String to a fileName
     * @param str String to translat
     * @return filename
     */
    String toFileName(String str);

    /**
     * Applies the given policy after this policy has been applied
     * @param otherPolicy other policy to apply
     * @return The combined policy
     */
    default FileNamePolicy andThen(FileNamePolicy otherPolicy) {
        return str -> otherPolicy.toFileName(toFileName(str));
    }

    /**
     * Applies the given policy before this policy has been applied
     * @param otherPolicy other policy to apply
     * @return The combined policy
     */
    default FileNamePolicy compose(FileNamePolicy otherPolicy) {
        return str -> toFileName(otherPolicy.toFileName(str));
    }

}
