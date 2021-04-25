$(document).ready(function(){
	var dropZone = $('#upload-container');

	$('#file-input').focus(function() {
		$('label').addClass('focus');
	})
	.focusout(function() {
		$('label').removeClass('focus');
	});


	dropZone.on('drag dragstart dragend dragover dragenter dragleave drop', function(){
		return false;
	});

	dropZone.on('dragover dragenter', function() {
		dropZone.addClass('dragover');
	});

	dropZone.on('dragleave', function(e) {
		dx = e.pageX - dropZone.offset().left;
		dy = e.pageY - dropZone.offset().top;
		if ((dx < 0) || (dx > dropZone.width()) || (dy < 0) || (dy > dropZone.height())) {
			dropZone.removeClass('dragover');
		}
	});

	dropZone.on('drop', function(e) {
		dropZone.removeClass('dragover');
		files = e.originalEvent.dataTransfer.files;
		sendFiles(files);
	});

	$('#file-input').change(function() {
		files = this.files;
		sendFiles(files);
	});


	function sendFiles(files) {
		// let maxFileSize = 5242880;
		var Data = new FormData();
		var files = $('#file')[0].files;
		// $(files).each(function(index, file) {
		// 	if ((file.size <= maxFileSize) && ((file.type == 'image/png') || (file.type == 'image/jpeg'))) {
		// 		Data.append('images[]', file);
		// 	}
		// });
		if (files.length > 0) {
			Data.append('file', files[0]);
			$.ajax({
				url: 'http://localhost:5000/uploader',
				type: 'POST',
				data: Data,
				contentType: false,
				processData: false,
				success: function(response){
					if(response !=0){
						alert('Successfully uploaded file')
					}else{
						alert('File not uploaded');
					}
				},
			});

		}else{
			alert("Please select a file.");
		}

		// $.ajax({
		// 	url: dropZone.attr('action'),
		// 	type: dropZone.attr('method'),
		// 	data: Data,
		// 	contentType: false,
		// 	processData: false,
		// 	success: function(data) {
		// 		alert ('Файлы были успешно загружены!');
		// 	}
		// });
	}
})