module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  plugins: [
    function ({ addUtilities }) {
      const newUtilities = {
        '.scrollbar-thin': {
          scrollbarWidth: '2px',
          // scrollbarColor: 'rgba(13, 83, 222, 1)',
          '&::-webkit-scrollbar': {
            width: '6px',
            height: '6px'
          },
          '&::-webkit-scrollbar-track': {
            backgroundColor: 'transparent'
          },
          '&::-webkit-scrollbar-thumb': {
            // backgroundColor: 'rgba(13, 83, 222, 0.01)',
            borderRadius: '20px',
            border: '3px solid transparent'
          },
          '&:hover::-webkit-scrollbar-thumb': {
            width: '6px',
            border: '3px solid rgb(229 231 235)',
            backgroundColor: 'rgb(229 231 235)'
          }
        }
        // 你可以添加更多自定义的滚动条样式
      };
      addUtilities(newUtilities, ['responsive', 'hover']);
    },
  ],
  // ...other configurations
}