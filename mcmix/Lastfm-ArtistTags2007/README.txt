The LastFM-ArtistTags2007 Data set
Version 1.0
June 2008


What is this?

    This is a set of artist tag data collected from Last.fm using
    the Audioscrobbler webservice during the spring of 2007.  

    The data consists of the raw tag counts for the 100 most
    frequently occuring tags that Last.fm listeners have applied
    to over 20,000 artists.

    An undocumented (and deprecated) option of the audioscrobbler
    web service was used to bypass the Last.fm normalization of tag
    counts.  This data set provides raw tag counts.

Data Format:

  The data is formatted one entry per line as follows:

  musicbrainz-artist-id<sep>artist-name<sep>tag-name<sep>raw-tag-count


Example:

    11eabe0c-2638-4808-92f9-1dbd9c453429<sep>Deerhoof<sep>american<sep>14
    11eabe0c-2638-4808-92f9-1dbd9c453429<sep>Deerhoof<sep>animals<sep>5
    11eabe0c-2638-4808-92f9-1dbd9c453429<sep>Deerhoof<sep>art punk<sep>21
    11eabe0c-2638-4808-92f9-1dbd9c453429<sep>Deerhoof<sep>art rock<sep>18
    11eabe0c-2638-4808-92f9-1dbd9c453429<sep>Deerhoof<sep>atmospheric<sep>4
    11eabe0c-2638-4808-92f9-1dbd9c453429<sep>Deerhoof<sep>avantgarde<sep>3

Data Statistics:

    Total Lines:      952810
    Unique Artists:    20907
    Unique Tags:      100784
    Total Tags:      7178442

Filtering:

    Some minor filtering has been applied to the tag data.  Last.fm will
    report tag with counts of zero or less on occasion. These tags have
    been removed. 
    
    Artists with no tags have not been included in this data set.
    Of the nearly quarter million artists that were inspected, 20,907
    artists had 1 or more tags.

Files:

    ArtistTags.dat  - the tag data
    README.txt      - this file
    artists.txt     - artists ordered by tag count
    tags.txt        - tags ordered by tag count

License:

    The data in LastFM-ArtistTags2007 is distributed with permission of
    Last.fm.  The data is made available for non-commercial use only under
    the Creative Commons Attribution-NonCommercial-ShareAlike UK License.
    Those interested in using the data or web services in a commercial
    context should contact partners at last dot fm. For more information 
    see http://www.audioscrobbler.net/data/

Acknowledgements:

    Thanks to Last.fm for providing the access to this tag data via their
    web services

Contact:

    This data was collected, filtered and by Paul Lamere of Sun Labs. Send
    questions or comments to Paul.Lamere@sun.com

