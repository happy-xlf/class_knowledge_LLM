document.addEventListener('DOMContentLoaded', function() {
  layui.use(['form', 'layer'], function(){
    var form = layui.form;
    var layer = layui.layer;
    var $ = layui.jquery;
    
    // 初始化表单
    form.render();
    
    // 配置marked选项
    marked.setOptions({
      renderer: new marked.Renderer(),
      highlight: function(code, lang) {
        if (lang && hljs.getLanguage(lang)) {
          return hljs.highlight(code, { language: lang }).value;
        }
        return hljs.highlightAuto(code).value;
      },
      pedantic: false,
      gfm: true,
      breaks: true,
      sanitize: false,
      smartypants: false,
      xhtml: false
    });
    
    // 导出PDF功能
    $('#exportPDF').on('click', function() {
      var element = document.getElementById('chatContainer');
      var opt = {
        margin: 1,
        filename: '教学方案.pdf',
        image: { type: 'jpeg', quality: 0.98 },
        html2canvas: { scale: 2 },
        jsPDF: { unit: 'in', format: 'a4', orientation: 'portrait' }
      };

      // 显示加载提示
      layer.load(1, {
        shade: [0.1,'#fff']
      });

      // 生成PDF
      html2pdf().set(opt).from(element).save().then(function() {
        layer.closeAll('loading');
        layer.msg('PDF导出成功');
      }).catch(function(err) {
        layer.closeAll('loading');
        layer.msg('PDF导出失败：' + err.message);
      });
    });
    
    // 监听表单提交
    form.on('submit(generatePlan)', function(data){
      // 显示加载中
      var loadingIndex = layer.load(1, {
        shade: [0.1, '#fff']
      });
      
      // 清空对话容器
      $('#chatContainer').empty();
      
      // 获取选中的课程名称
      var courseName = $('#courseSelect option:selected').text();
      if (!courseName || courseName === '请选择课程') {
        layer.close(loadingIndex);
        layer.msg('请选择课程', {icon: 2});
        return false;
      }
      
      // 添加用户消息
      var userMessage = $('<div class="message user-message"></div>');
      userMessage.html('<strong>用户:</strong><br>' + 
                       '课程: ' + courseName + '<br>' +
                       '关键词: ' + $('#keywordsInput').val() + '<br>' +
                       '教案文本: ' + $('#contentTextarea').val());
      $('#chatContainer').append(userMessage);
      
      // 添加助手消息占位
      var assistantMessage = $('<div class="message assistant-message"></div>');
      assistantMessage.html('<strong>助手:</strong><br><div id="assistantResponse"></div>');
      $('#chatContainer').append(assistantMessage);
      
      // 添加打字指示器
      var typingIndicator = $('<div class="typing-indicator"><span></span><span></span><span></span></div>');
      $('#assistantResponse').append(typingIndicator);
      
      // 滚动到底部
      $('#chatContainer').scrollTop($('#chatContainer')[0].scrollHeight);
      
      // 创建一个隐藏的textarea来存储Markdown内容
      var markdownTextarea = $('<textarea id="markdownText" style="display:none"></textarea>');
      $('#assistantResponse').append(markdownTextarea);
      
      // 创建一个div来显示解析后的HTML
      var markdownContainer = $('<div id="markdownContainer" class="markdown-body"></div>');
      $('#assistantResponse').append(markdownContainer);
      
      // 使用Fetch API处理流式响应
      fetch('/api/generate_teaching_plan', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(data.field)
      }).then(response => {
        // 获取响应的可读流
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let markdownContent = '';
        
        // 移除打字指示器
        $('.typing-indicator').remove();
        
        // 关闭加载提示
        layer.close(loadingIndex);
        
        // 处理流式数据
        function processStream({ done, value }) {
          if (done) {
            document.getElementById('markdownContainer').innerHTML = marked.parse(markdownContent);
            
            // 添加导出按钮
            var exportButtons = $('<div class="export-buttons"></div>');
            exportButtons.append('<button class="layui-btn layui-btn-sm" onclick="exportMarkdown()">导出Markdown</button>');
            exportButtons.append('<button class="layui-btn layui-btn-sm layui-btn-normal" onclick="exportPDF()">导出PDF</button>');
            $('#assistantResponse').append(exportButtons);
            
            // 滚动到底部
            $('#chatContainer').scrollTop($('#chatContainer')[0].scrollHeight);
            return;
          }
          
          // 解码二进制数据
          const chunk = decoder.decode(value, { stream: true });
          markdownContent += chunk;
          
          // 更新textarea和Markdown显示
          $('#markdownText').val(markdownContent);
          
          // 使用marked.js解析Markdown并显示为HTML
          console.log(markdownContent);
          console.log(marked.parse(markdownContent));
          console.log("================");
          document.getElementById('markdownContainer').innerHTML = marked.parse(markdownContent);
          
          // 滚动到底部
          $('#chatContainer').scrollTop($('#chatContainer')[0].scrollHeight);
          
          // 继续读取流
          return reader.read().then(processStream);
        }
        
        // 开始读取流
        return reader.read().then(processStream);
      }).catch(error => {
        // 关闭加载提示
        layer.close(loadingIndex);
        
        // 移除打字指示器
        $('.typing-indicator').remove();
        
        // 显示错误信息
        $('#assistantResponse').html('<span style="color: red;">请求失败，请稍后重试</span>');
        
        // 滚动到底部
        $('#chatContainer').scrollTop($('#chatContainer')[0].scrollHeight);
        
        console.error('Error:', error);
      });
      
      return false; // 阻止表单默认提交
    });
  });
});

// 导出Markdown函数
window.exportMarkdown = function() {
  var markdownContent = $('#markdownText').val();
  var courseName = $('#courseSelect option:selected').text();
  var fileName = courseName + '_教学方案_' + new Date().toISOString().slice(0, 10) + '.md';
  
  // 创建Blob对象
  var blob = new Blob([markdownContent], { type: 'text/markdown' });
  
  // 创建下载链接
  var a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = fileName;
  
  // 触发下载
  document.body.appendChild(a);
  a.click();
  
  // 清理
  setTimeout(function() {
    document.body.removeChild(a);
    window.URL.revokeObjectURL(a.href);
  }, 0);
};

// 导出PDF函数
window.exportPDF = function() {
  // 创建一个临时容器，只包含教学方案内容
  var tempContainer = document.createElement('div');
  tempContainer.style.padding = '20px';
  tempContainer.style.fontFamily = 'Arial, sans-serif';
  
  // 添加标题
  var title = document.createElement('h1');
  title.textContent = $('#courseSelect option:selected').text() + ' 教学方案';
  title.style.textAlign = 'center';
  title.style.marginBottom = '20px';
  tempContainer.appendChild(title);
  
  // 添加日期
  var date = document.createElement('p');
  date.textContent = '生成日期: ' + new Date().toLocaleDateString();
  date.style.textAlign = 'right';
  date.style.marginBottom = '20px';
  tempContainer.appendChild(date);
  
  // 复制教学方案内容
  var content = document.createElement('div');
  content.innerHTML = $('#markdownContainer').html();
  // 输出台打印
//   console.log(content.innerHTML);
  tempContainer.appendChild(content);
  
  // 设置PDF选项
  var opt = {
    margin: 1,
    filename: $('#courseSelect option:selected').text() + '_教学方案.pdf',
    image: { type: 'jpeg', quality: 0.98 },
    html2canvas: { scale: 2 },
    jsPDF: { unit: 'in', format: 'a4', orientation: 'portrait' }
  };

  // 显示加载提示
  layer.load(1, {
    shade: [0.1,'#fff']
  });

  // 生成PDF
  html2pdf().set(opt).from(tempContainer).save().then(function() {
    layer.closeAll('loading');
    layer.msg('PDF导出成功');
  }).catch(function(err) {
    layer.closeAll('loading');
    layer.msg('PDF导出失败：' + err.message);
  });
}; 