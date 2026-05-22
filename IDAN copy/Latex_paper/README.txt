

July 12, 2017

IEEE_lsens.cls is the official LaTeX class for authors of the IEEE Sensors
Letters (LSENS) papers. See the LSENS website:

http://www.ieee-sensors.org/

for the latest version and other LSENS author support information.

For LaTeX technical support on the use of IEEE_lsens.cls, see:

http://www.michaelshell.org/contact.html


IEEE_lsens.cls is derived from IEEEtran.cls, but there are a few notable
differences. The IEEE_lsens_HOWTO.pdf covers these differences, as well
as any special rules/aspects of the LSENS authoring and submission process,
and should be read carefully by all IEEE_lsens.cls users.

In this way, IEEE authors who are already familar with IEEEtran can quickly
learn the unique features and behavior of the IEEE_lsens LaTeX class simply
by reading the relatively short IEEE_lsens_HOWTO.pdf.

Those authors who are not familar with LaTeX work under IEEEtran should
also read the full IEEEtran user manual, IEEEtran_HOWTO.pdf, which can be
obtained at the journal author support section within the main IEEE website
at http://www.ieee.org/ or at the IEEEtran package distribution page on
the CTAN site at http://www.ctan.org/pkg/ieeetran .


*** Existing IEEEtran.cls documents will require a few changes, especially
    with regard to the format of \author{} and other title page setup,
    in order to work properly under IEEE_lsens.cls. ***
   
*** The standard IEEE BibTeX style (IEEEtran.bst) will work fine as-is for
    bibliography/reference formatting of LSENS papers. ***


See the attached bare_lsens.tex for an example "bare bones" LSENS paper
starter file.


Best wishes for all your publication endeavors,

Michael Shell
http://www.michaelshell.org/

********************************** Files **********************************

README.txt             - This file.

IEEE_lsens_HOWTO.pdf   - The IEEE_lsens LaTeX class user manual.

IEEE_lsens.cls         - The IEEEtran_lsens LaTeX class file.

bare_lsens.tex         - A bare bones starter file for LSENS papers.

changelog.txt          - The revision history.


*************************************************************************
 Legal Notice:
 This code is offered as-is without any warranty either expressed or
 implied; without even the implied warranty of MERCHANTABILITY or
 FITNESS FOR A PARTICULAR PURPOSE! 
 User assumes all risk.
 In no event shall IEEE or any contributor to this code be liable for
 any damages or losses, including, but not limited to, incidental,
 consequential, or any other damages, resulting from the use or misuse
 of any information contained here.

 All comments are the opinions of their respective authors and are not
 necessarily endorsed by the IEEE.

 This work is distributed under the LaTeX Project Public License (LPPL)
 ( http://www.latex-project.org/ ) version 1.3, and may be freely used,
 distributed and modified. A copy of the LPPL, version 1.3, is included
 in the base LaTeX documentation of all distributions of LaTeX released
 2003/12/01 or later.
 Retain all contribution notices and credits.
 ** Modified files should be clearly indicated as such, including  **
 ** renaming them and changing author support contact information. **

*************************************************************************