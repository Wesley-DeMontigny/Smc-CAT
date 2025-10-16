#ifndef MSG_HPP
#define MSG_HPP

#include <string>


/**
 * @brief Basic class for outputting errors and warnings to the screen
 * 
 */
namespace Msg {
   void   error(std::string s);        // Output an error to the screen (exits the program)
   void   warning(std::string s);      // Output a warning to the screen
}

#endif
